# infer_lisat_eval.py
import argparse, os, sys, pathlib
import numpy as np, cv2, torch
from PIL import Image

# --- make the repo importable regardless of where you run from ---
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- use LISAT_eval (no SESAME imports) ---
from model.LISAT_eval import load_pretrained_model_LISAT
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad
from utils import prepare_input


def _safe_save_png(path, mask_array):
    # Ensure 8-bit single-channel PNG
    if mask_array.dtype != np.uint8:
        mask_array = mask_array.astype(np.uint8)
    if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        mask_array = mask_array[..., 0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(mask_array, mode="L").save(path, format="PNG")


def _load_model(model_path: str):
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype, tag = torch.bfloat16, "bf16"
    elif torch.cuda.is_available():
        dtype, tag = torch.float16, "fp16"
    else:
        dtype, tag = torch.float32, "fp32"

    tokenizer, model, vision_tower, _ = load_pretrained_model_LISAT(
        model_path=model_path,
        device_map="auto",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # IMPORTANT: load in the dtype you want (see next snippet)
        torch_dtype=dtype,  # <-- only works if loader supports it
    )

    # If the loader used accelerate dispatch/offload, DON'T .to() it.
    # (If you really want, you can still cast vision_tower if it's NOT dispatched.)
    try:
        vision_tower = vision_tower.to(dtype=dtype)
    except Exception:
        pass

    tokenizer.padding_side = "left"
    return tokenizer, model, vision_tower, tag


@torch.inference_mode()
def predict_mask(
    image_path: str,
    prompt: str,
    model_path: str = "checkpoints/LISAt-7b",   # or "jquenum/LISAt-7b"
    image_size: int = 1024,
    max_new_tokens: int = 512,
    out_mask_path: str = "outputs/mask.png",
):
    tokenizer, model, vision_tower, dtype_tag = _load_model(model_path)

    # Preprocess image the repo's way
    img_processor = ImageProcessor(vision_tower.image_processor, image_size)
    image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_path)

    # Conversation prompt (LLaVA style, with an image token)
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    conversation_list = [conv.get_prompt()]

    # Replace tokens if the model expects image start/end tokens
    if getattr(model.config, "mm_use_im_start_end", False):
        conversation_list = replace_image_tokens(conversation_list)

    input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding="left")

    # Pack and move to device/dtype
    inputs = {
        "image_path": image_path,
        "images_clip": torch.stack([image_clip], dim=0),
        "images": torch.stack([image], dim=0),
        "input_ids": input_ids,
        "sam_mask_shape_list": [sam_mask_shape],
    }
    inputs = prepare_input(inputs, dtype_tag, is_cuda=torch.cuda.is_available())

    # Forward using LISAT_eval's built-in evaluate()
    output_ids, pred_masks, object_presence = model.evaluate(
        inputs["images_clip"],
        inputs["images"],
        inputs["input_ids"],
        inputs["sam_mask_shape_list"],
        max_new_tokens=max_new_tokens,
    )

    # Strict binary mask: background=0 (black), object=255 (white)
    pm = pred_masks[0]
    if pm.dim() == 3:
        pm = pm[0]
    mask = (pm.detach().float().cpu().numpy() > 0).astype(np.uint8) * 255

    # Save robustly (PNG)
    if not out_mask_path.lower().endswith(".png"):
        out_mask_path = os.path.splitext(out_mask_path)[0] + ".png"
    _safe_save_png(out_mask_path, mask)

    # Optional: decode generated text
    real_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    text = tokenizer.batch_decode(real_ids, skip_special_tokens=True)[0] if real_ids.numel() else ""

    return {"mask_path": out_mask_path, "object_present": bool(object_presence[0]), "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model_path", default="checkpoints/LISAt-7b")  # or "jquenum/LISAt-7b"
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", default="outputs/mask.png")
    args = ap.parse_args()

    res = predict_mask(
        image_path=args.image,
        prompt=args.prompt,
        model_path=args.model_path,
        image_size=args.image_size,
        max_new_tokens=args.max_new_tokens,
        out_mask_path=args.out,
    )
    print(f"[OK] mask saved to: {res['mask_path']}")
    print(f"object_present={res['object_present']}")
    if res['text']:
        print(f"text: {res['text']}")


if __name__ == "__main__":
    main()