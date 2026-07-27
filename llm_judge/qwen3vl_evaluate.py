#!/usr/bin/env python3
"""
Self-contained Qwen3-VL evaluator for revised visual quality and scene realism.

The evaluator asks the LLM for only the two revised scores:
  - visual_quality
  - scene_realism

It carries the old policy-preservation score from the input file into the
output CSV and computes llm_judge_avg = mean(policy_preservation,
visual_quality, scene_realism). It does not re-score policy preservation.

Examples:
  python llm_judge/qwen3vl_evaluate.py --print_prompt

  OPENROUTER_API_KEY=... python llm_judge/qwen3vl_evaluate.py \
    --inputs benchmark_pairs.xlsx \
    --exclude_sheets merged_source_debug \
    --dedupe_methods 'NB2.5 (ZS)' DIFFusion \
    --out llm_judge_results_qwen3vl.csv \
    --workers 1 --max_retries 8

The defaults select OpenRouter's chat endpoint and
qwen/qwen3-vl-32b-instruct. OpenAI Responses and other models remain available
through explicit command-line flags.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import glob
import io
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Iterable, Optional, Sequence, Tuple
import urllib.error
import urllib.request
import zipfile
from xml.etree import ElementTree as ET


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
SCORE_ENUM = list(range(1, 11))
RAW_FAILURE_DEBUG_CHARS = 12000

JUDGE_DEFINITION = """Score exactly these two revised criteria.

1) visual_quality:
Score whether the edited/output image looks like a natural, authentic image in
the same capture modality as the original.

For street-view or ground-level images, it should look like a real GSV/photo
capture rather than a cartoon, illustration, CGI render, painting, collage, or
oddly formatted/generated image. Penalize synthetic image style, over-smoothed
AI texture, strange color processing, low sharpness, seams, warping, ghosting,
garbled texture, melted objects, obvious compositing artifacts, or any visual
presentation that makes the image itself feel unnatural.

For satellite or aerial images, it should look like a real remote-sensing image
with plausible sensor texture, scale, resolution, shadows, land-cover detail,
and image formatting. Penalize painterly/CGI appearance, fake map-like styling,
unnatural texture, inconsistent resolution, or obvious generation artifacts.

This criterion is about the image as an image, not whether the edited world is a
good intervention.

2) scene_realism:
Score whether the edited world is physically, semantically, and contextually
plausible in this specific scene.

Objects should have plausible function, material, scale, geometry, attachment,
and spatial relationships. Doors should plausibly open and close. Windows,
porches, roads, sidewalks, wires, roofs, trees, parking areas, and buildings
should fit the surrounding structure and land use. A brick building should not
become materially incoherent. Roads and infrastructure should connect logically.

Penalize edits that appear to game the target metric by inserting context-
breaking or excessive features: many porches in a neighborhood where they do not
fit, a helicopter pad added to make an area look greener, luxury objects pasted
into a setting where they make no physical or social sense, repeated objects
that would not actually exist there, or changes that conflict with the local
neighborhood/building/satellite context.

Do not score strict policy preservation here unless the violation also makes
the edited world physically or contextually implausible. The old
policy_preservation score remains separate and is copied from the input file.
"""

JUDGE_PROMPT = f"""You are a strict, consistent image-edit evaluator.
You receive two images: first the original/reference image, then the edited/output image.

Your task is to produce ONLY two integer scores:
- visual_quality
- scene_realism

{JUDGE_DEFINITION.strip()}

Scoring:
- Use a 1-10 integer scale.
- 10 means excellent, natural, and plausible.
- 1 means severe failure.
- Return valid JSON only, with exactly these keys:
  {{"visual_quality": <int>, "scene_realism": <int>}}
"""

INPUT_HEADERS = {"input path", "input_path", "input", "source image", "source_path"}
OUTPUT_HEADERS = {"output path", "output_path", "output", "edited image", "edited_path"}
HEADER_ALIASES = {
    "task": ["task", "metric"],
    "method": ["method", "model"],
    "policy_preservation": [
        "policy preservation",
        "policy_preservation",
        "llm reference preservation",
        "llm_reference_preservation",
        "llm policy preservation",
        "llm_policy_preservation",
        "old_policy_preservation",
    ],
}

OUTPUT_FIELDS = [
    "source_file", "sheet", "row_number", "task", "method",
    "input_path", "output_path",
    "policy_preservation", "visual_quality", "scene_realism", "llm_judge_avg",
    "judge_provider", "judge_model", "judge_backend",
    "input_tokens", "output_tokens", "total_tokens", "error",
]


class JudgeRawResponseError(RuntimeError):
    def __init__(self, message: str, raw_text: str = "", body: Optional[dict] = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.body = body


def norm_header(value) -> str:
    return str(value or "").strip().lower().replace("\n", " ").replace("_", " ")


def display(value) -> str:
    return "" if value is None else str(value)


def col_to_index(col: str) -> int:
    out = 0
    for ch in col:
        if "A" <= ch <= "Z":
            out = out * 26 + ord(ch) - 64
    return out


def split_ref(ref: str) -> Tuple[int, int]:
    m = re.match(r"([A-Z]+)([0-9]+)", ref or "")
    if not m:
        return 0, 0
    return int(m.group(2)), col_to_index(m.group(1))


def xml_text(node: ET.Element) -> str:
    return "".join(node.itertext())


def parse_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [xml_text(si) for si in root if si.tag.endswith("si")]


def norm_target(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return str(PurePosixPath("xl") / target)


def workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rels = {}
    rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    for rel in rel_root:
        if rel.tag.startswith(rel_ns):
            rels[rel.attrib.get("Id", "")] = norm_target(rel.attrib.get("Target", ""))
    out = []
    rid_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    for elem in wb_root.iter():
        if elem.tag.endswith("sheet"):
            name = elem.attrib.get("name", "")
            path = rels.get(elem.attrib.get(rid_key, ""))
            if name and path:
                out.append((name, path))
    return out


def parse_sheet(zf: zipfile.ZipFile, path: str, shared: Sequence[str]) -> dict[int, dict[int, str]]:
    root = ET.fromstring(zf.read(path))
    rows: dict[int, dict[int, str]] = {}
    for cell in root.iter():
        if not cell.tag.endswith("c"):
            continue
        row_idx, col_idx = split_ref(cell.attrib.get("r", ""))
        if row_idx <= 0 or col_idx <= 0:
            continue
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            value = xml_text(cell)
        else:
            value = ""
            for child in cell:
                if child.tag.endswith("v") and child.text is not None:
                    raw = child.text
                    if cell_type == "s":
                        try:
                            value = shared[int(raw)]
                        except Exception:
                            value = raw
                    else:
                        value = raw
                    break
        rows.setdefault(row_idx, {})[col_idx] = value
    return rows


def first_col(header_map: dict[str, int], aliases: Iterable[str]) -> Optional[int]:
    for alias in aliases:
        key = norm_header(alias)
        if key in header_map:
            return header_map[key]
    return None


def find_header(cells: dict[int, dict[int, str]]) -> tuple[Optional[int], dict[str, int]]:
    for row_idx in range(1, 13):
        row = cells.get(row_idx, {})
        header_map = {norm_header(v): col for col, v in row.items() if display(v).strip()}
        if any(h in header_map for h in INPUT_HEADERS) and any(h in header_map for h in OUTPUT_HEADERS):
            return row_idx, header_map
    return None, {}


def row_meta(row: dict[int, str], header_map: dict[str, int], field: str) -> str:
    col = first_col(header_map, HEADER_ALIASES[field])
    return display(row.get(col)).strip() if col else ""


def iter_xlsx_rows(path: Path, exclude_sheets: set[str]) -> Iterable[dict]:
    with zipfile.ZipFile(path) as zf:
        shared = parse_shared_strings(zf)
        for sheet_name, sheet_path in workbook_sheets(zf):
            if sheet_name.lower() in exclude_sheets or "summary" in sheet_name.lower():
                continue
            cells = parse_sheet(zf, sheet_path, shared)
            header_row, header_map = find_header(cells)
            if header_row is None:
                continue
            input_col = first_col(header_map, INPUT_HEADERS)
            output_col = first_col(header_map, OUTPUT_HEADERS)
            if not input_col or not output_col:
                continue
            for row_idx in sorted(k for k in cells if k > header_row):
                row = cells[row_idx]
                inp = display(row.get(input_col)).strip()
                outp = display(row.get(output_col)).strip()
                if not inp or not outp:
                    continue
                yield {
                    "source_file": str(path),
                    "sheet": sheet_name,
                    "row_number": str(row_idx),
                    "input_path": inp,
                    "output_path": outp,
                    "task": row_meta(row, header_map, "task"),
                    "method": row_meta(row, header_map, "method"),
                    "policy_preservation": row_meta(row, header_map, "policy_preservation"),
                }


def iter_csv_rows(path: Path) -> Iterable[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        norm_to_header = {norm_header(h): h for h in headers}
        input_header = next((norm_to_header[h] for h in INPUT_HEADERS if h in norm_to_header), None)
        output_header = next((norm_to_header[h] for h in OUTPUT_HEADERS if h in norm_to_header), None)
        if not input_header or not output_header:
            return
        for idx, row in enumerate(reader, 2):
            yield {
                "source_file": str(path),
                "sheet": "",
                "row_number": str(idx),
                "input_path": display(row.get(input_header)).strip(),
                "output_path": display(row.get(output_header)).strip(),
                "task": display(row.get(next((norm_to_header[norm_header(a)] for a in HEADER_ALIASES["task"] if norm_header(a) in norm_to_header), ""), "")),
                "method": display(row.get(next((norm_to_header[norm_header(a)] for a in HEADER_ALIASES["method"] if norm_header(a) in norm_to_header), ""), "")),
                "policy_preservation": display(row.get(next((norm_to_header[norm_header(a)] for a in HEADER_ALIASES["policy_preservation"] if norm_header(a) in norm_to_header), ""), "")),
            }


def iter_input_rows(inputs: Sequence[str], exclude_sheets: Sequence[str]) -> list[dict]:
    rows = []
    excluded = {s.lower() for s in exclude_sheets}
    for raw in inputs:
        for expanded in sorted(glob.glob(os.path.expanduser(raw))) or [raw]:
            path = Path(expanded).expanduser()
            if path.suffix.lower() == ".xlsx":
                rows.extend(iter_xlsx_rows(path, excluded))
            elif path.suffix.lower() == ".csv":
                rows.extend(iter_csv_rows(path))
            else:
                print(f"[WARN] unsupported input file: {path}", file=sys.stderr)
    return rows


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (row["source_file"], row["sheet"], row["row_number"], row["output_path"])


def pair_key(row: dict) -> tuple[str, str]:
    return (row["input_path"], row["output_path"])


def dedupe_rows(rows: Sequence[dict], dedupe_methods: Optional[Sequence[str]]) -> list[dict]:
    methods = None if dedupe_methods is None else set(dedupe_methods)
    seen = set()
    out = []
    for row in rows:
        should = methods is None or row.get("method") in methods
        key = pair_key(row)
        if should and key in seen:
            continue
        if should:
            seen.add(key)
        out.append(row)
    return out


def maybe_resize(raw: bytes, max_side: int) -> bytes:
    if not max_side:
        return raw
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        return raw


def data_url(path: str, max_side: int) -> str:
    raw = Path(path).expanduser().read_bytes()
    raw = maybe_resize(raw, max_side)
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["visual_quality", "scene_realism"],
        "properties": {
            "visual_quality": {"type": "integer", "enum": SCORE_ENUM},
            "scene_realism": {"type": "integer", "enum": SCORE_ENUM},
        },
    }


def post_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise JudgeRawResponseError(f"HTTP response was not JSON: {e}", raw_text=raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise JudgeRawResponseError(f"HTTP {e.code}: {body[:1000]}", raw_text=body)


def parse_json_text(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                for key in ("text", "content"):
                    if isinstance(block.get(key), str):
                        chunks.append(block[key])
                        break
        return "\n".join(chunks)
    return ""


def extract_chat_json(body: dict) -> Optional[dict]:
    message = ((body.get("choices") or [{}])[0].get("message") or {})
    candidates = [content_to_text(message.get("content"))]
    if isinstance(message.get("reasoning"), str):
        candidates.append(message["reasoning"])
    for detail in message.get("reasoning_details") or []:
        if isinstance(detail, dict) and isinstance(detail.get("text"), str):
            candidates.append(detail["text"])
    for candidate in candidates:
        parsed = parse_json_text(candidate)
        if parsed is not None:
            return parsed
    return None


def response_text_candidates(body: dict) -> list[str]:
    candidates = []
    if isinstance(body.get("output_text"), str):
        candidates.append(body["output_text"])
    for item in body.get("output", []) or []:
        if isinstance(item.get("reasoning"), str):
            candidates.append(item["reasoning"])
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                candidates.append(part["text"])
            elif isinstance(part, dict) and isinstance(part.get("reasoning"), str):
                    candidates.append(part["reasoning"])
    return candidates


def extract_responses_json(body: dict) -> Optional[dict]:
    for candidate in response_text_candidates(body):
        parsed = parse_json_text(candidate)
        if parsed is not None:
            return parsed
    return None


def usage(body: dict) -> tuple[int, int, int]:
    u = body.get("usage") or {}
    inp = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
    out = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
    return inp, out, int(u.get("total_tokens") or inp + out)


def parse_scores(parsed: Optional[dict]) -> dict:
    if parsed is None:
        raise RuntimeError("empty or unparseable judge response")
    vq = parsed.get("visual_quality", parsed.get("revised_visual_quality"))
    sr = parsed.get("scene_realism", parsed.get("revised_scene_realism"))
    scores = {"visual_quality": int(vq), "scene_realism": int(sr)}
    for key, value in scores.items():
        if value < 1 or value > 10:
            raise RuntimeError(f"{key} out of range: {value}")
    return scores


def raw_response_text(exc: BaseException) -> str:
    if not isinstance(exc, JudgeRawResponseError):
        return ""
    if exc.raw_text:
        return exc.raw_text
    if exc.body is not None:
        try:
            return json.dumps(exc.body, ensure_ascii=False, indent=2)
        except Exception:
            return str(exc.body)
    return ""


def log_final_failure(row: dict, error: str, exc: BaseException) -> None:
    raw = raw_response_text(exc)
    if len(raw) > RAW_FAILURE_DEBUG_CHARS:
        raw = raw[:RAW_FAILURE_DEBUG_CHARS] + "\n...[truncated]..."
    print(
        "\n=== RAW_JUDGE_FAILURE_BEGIN ===\n"
        f"source_file: {row.get('source_file', '')}\n"
        f"sheet: {row.get('sheet', '')}\n"
        f"row_number: {row.get('row_number', '')}\n"
        f"task: {row.get('task', '')}\n"
        f"method: {row.get('method', '')}\n"
        f"input_path: {row.get('input_path', '')}\n"
        f"output_path: {row.get('output_path', '')}\n"
        f"error: {error}\n"
        "--- raw_response ---\n"
        f"{raw if raw else '[no raw response captured]'}\n"
        "=== RAW_JUDGE_FAILURE_END ===\n",
        file=sys.stderr,
        flush=True,
    )


def judge(row: dict, args, api_key: str) -> tuple[Optional[dict], tuple[int, int, int], str]:
    last_error = ""
    last_exc: Optional[BaseException] = None
    for attempt in range(args.max_retries + 1):
        try:
            if args.backend == "chat":
                body = post_json(OPENROUTER_CHAT_URL, {
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": JUDGE_PROMPT},
                        {"type": "text", "text": "Original/reference image:"},
                        {"type": "image_url", "image_url": {"url": data_url(row["input_path"], args.max_side)}},
                        {"type": "text", "text": "Edited/output image to judge:"},
                        {"type": "image_url", "image_url": {"url": data_url(row["output_path"], args.max_side)}},
                    ]}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "judgment", "strict": True, "schema": schema()}},
                }, api_key, args.timeout)
                try:
                    return parse_scores(extract_chat_json(body)), usage(body), ""
                except Exception as e:
                    raise JudgeRawResponseError(str(e), body=body)
            url = OPENAI_RESPONSES_URL if args.provider == "openai" else OPENROUTER_RESPONSES_URL
            body = post_json(url, {
                "model": args.model,
                "temperature": args.temperature,
                "input": [
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": "Original/reference image:"},
                        {"type": "input_image", "image_url": data_url(row["input_path"], args.max_side)},
                        {"type": "input_text", "text": "Edited/output image to judge:"},
                        {"type": "input_image", "image_url": data_url(row["output_path"], args.max_side)},
                    ]},
                ],
                "text": {"format": {"type": "json_schema", "name": "judgment", "strict": True, "schema": schema()}},
            }, api_key, args.timeout)
            try:
                return parse_scores(extract_responses_json(body)), usage(body), ""
            except Exception as e:
                raise JudgeRawResponseError(str(e), body=body)
        except Exception as e:
            last_exc = e
            last_error = str(e)
            if attempt < args.max_retries:
                time.sleep(min(60, 1.5 * (attempt + 1)))
    if last_exc is not None:
        log_final_failure(row, last_error, last_exc)
    return None, (0, 0, 0), last_error


def output_row(row: dict, args, scores: Optional[dict], token_usage: tuple[int, int, int], error: str) -> dict:
    policy = row.get("policy_preservation", "")
    vq = "" if scores is None else scores["visual_quality"]
    sr = "" if scores is None else scores["scene_realism"]
    try:
        avg = (float(policy) + float(vq) + float(sr)) / 3.0
    except Exception:
        avg = ""
    inp, out, total = token_usage
    return {
        **{k: row.get(k, "") for k in ["source_file", "sheet", "row_number", "task", "method", "input_path", "output_path"]},
        "policy_preservation": policy,
        "visual_quality": vq,
        "scene_realism": sr,
        "llm_judge_avg": avg,
        "judge_provider": args.provider,
        "judge_model": args.model,
        "judge_backend": args.backend,
        "input_tokens": inp or "",
        "output_tokens": out or "",
        "total_tokens": total or "",
        "error": error,
    }


def load_done(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()
    with open(path, newline="") as f:
        return {
            (r.get("source_file", ""), r.get("sheet", ""), r.get("row_number", ""), r.get("output_path", ""))
            for r in csv.DictReader(f)
            if r.get("visual_quality") and r.get("scene_realism") and not r.get("error")
        }


def append_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.is_file()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help="Input .xlsx/.csv files")
    ap.add_argument("--out", default="llm_judge_results_qwen3vl.csv")
    ap.add_argument("--provider", choices=["openai", "openrouter"], default="openrouter")
    ap.add_argument("--backend", choices=["responses", "chat"], default="chat")
    ap.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    ap.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    ap.add_argument("--exclude_sheets", nargs="+", default=[])
    ap.add_argument("--dedupe_methods", nargs="+", default=None, help="Only dedupe these method labels by input/output pair")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--print_prompt", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.print_prompt:
        print(JUDGE_PROMPT)
        return
    if not args.inputs:
        sys.exit("--inputs is required unless --print_prompt is used")
    api_key = os.getenv(args.api_key_env)
    if not api_key and not args.dry_run:
        sys.exit(f"{args.api_key_env} not set")

    rows = dedupe_rows(iter_input_rows(args.inputs, args.exclude_sheets), args.dedupe_methods)
    if args.limit is not None:
        rows = rows[:args.limit]
    out_path = Path(args.out)
    done = load_done(out_path)
    pending = [r for r in rows if row_key(r) not in done]
    print(f"[INFO] rows={len(rows)} done={len(done)} pending={len(pending)} workers={args.workers}", flush=True)
    if args.dry_run:
        for row in pending[:20]:
            print(row)
        return

    def run_one(row: dict) -> dict:
        if not Path(row["input_path"]).expanduser().is_file() or not Path(row["output_path"]).expanduser().is_file():
            return output_row(row, args, None, (0, 0, 0), "missing input or output image")
        scores, token_usage, error = judge(row, args, api_key)
        return output_row(row, args, scores, token_usage, error)

    if args.workers <= 1:
        for idx, row in enumerate(pending, 1):
            result = run_one(row)
            append_rows(out_path, [result])
            print(f"[{idx}/{len(pending)}] {result['method']} {result['task']} error={result['error']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, row): row for row in pending}
            for idx, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                append_rows(out_path, [result])
                print(f"[{idx}/{len(pending)}] {result['method']} {result['task']} error={result['error']}", flush=True)


if __name__ == "__main__":
    main()
