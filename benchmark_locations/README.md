# Benchmark Locations and Viewing Directions

These tab-separated files identify the **800 benchmark images**, split into
eight metrics with **100 images per metric**. Image files are not included.
Use `image_name` within the corresponding metric folder to match a record to
an image: `benchmark_images/<metric>/<image_name>`.

| Metric | Modality | Records | Image dimensions | File |
|---|---|---:|---|---|
| safety | Street View | 100 | 512 × 512 | [safety.tsv](safety.tsv) |
| lively | Street View | 100 | 512 × 512 | [lively.tsv](lively.tsv) |
| beautiful | Street View | 100 | 512 × 512 | [beautiful.tsv](beautiful.tsv) |
| wealthy | Street View | 100 | 512 × 512 | [wealthy.tsv](wealthy.tsv) |
| boring | Street View | 100 | 512 × 512 | [boring.tsv](boring.tsv) |
| depressing | Street View | 100 | 512 × 512 | [depressing.tsv](depressing.tsv) |
| greenery | Satellite | 100 | 1024 × 1024 | [greenery.tsv](greenery.tsv) |
| road_risk | Satellite | 100 | 1024 × 1024 | [road_risk.tsv](road_risk.tsv) |

`road_risk` corresponds to the pipeline domain `road_safety`.
The 58 replacement images in the local `extra_images/` directory are excluded.
The TSVs describe the eight benchmark folders, including their reference
images; they do not describe the separate experiment/replacement selection.

## Columns

| Column | Meaning |
|---|---|
| `image_name` | Exact benchmark filename, including extension. |
| `latitude` | Recorded latitude in decimal degrees; north is positive. |
| `longitude` | Recorded longitude in decimal degrees; east is positive. |
| `heading_degrees` | Street View camera compass direction: 0° north, 90° east, 180° south, 270° west. Blank for satellite images. |
| `panorama_id` | Original recorded Street View panorama ID where available; blank when unrecorded or inapplicable. |
| `width_px`, `height_px` | Decoded benchmark image dimensions, after any original preprocessing. |

Files are UTF-8 with a header row and literal tab separators. Empty cells mean
unrecorded or inapplicable values, never zero. Coordinate and heading precision
is preserved from the source records; precision is not a claim of positional
accuracy.

## Metadata provenance

- **542 Street View images:** coordinates, headings, and panorama IDs come
  from the local selection `benchmark_manifest.csv`.
- **58 Street View reference images:** coordinates come from that same
  manifest. Its headings and panorama IDs are blank, so headings are retained
  from the [historical repository manifest](https://github.com/HosamGen/VIDA-GEO/blob/b45c180afa8dc592863d177aea1661066609ed73/benchmark_images/benchmark_manifest.csv),
  which recorded recovery from original legacy filenames.
  The legacy source files were unavailable for a new independent recovery;
  no panorama IDs have been inferred for these images.
- **200 satellite images:** coordinates are read from the final latitude and
  longitude components of each filename and agree with the historical
  manifest. Street View headings do not apply to these images.

All 800 filenames match the historical manifest exactly. For the 542 directly
recorded headings and all 600 Street View coordinates, current and historical
metadata agree. Dimensions were measured from the actual benchmark images.

## Reacquiring images

For Street View, use the recorded panorama ID where still available, or the
latitude/longitude, together with `heading_degrees`. Heading is the horizontal
compass direction; pitch is the separate up/down angle. Field of view, requested
image size, and subsequent cropping also affect the resulting image. These
acquisition/preprocessing settings were not verified for every benchmark image
and are not supplied as assumed defaults. Google also refreshes imagery and
panorama IDs, so coordinates and headings alone cannot guarantee the original
pixels. See the [Street View request parameters](https://developers.google.com/maps/documentation/streetview/request-streetview).

For satellite images, coordinates and the final 1024 × 1024 dimensions are
available. The original provider, imagery date, zoom/ground resolution, and crop
parameters were not found in the supplied metadata. Those settings are needed
to reproduce the same tile; a Street View heading does not supply them.

All 800 originals decode as JPEG, including the
200 satellite files named with a `.png` extension; the TSVs preserve the exact
original filenames.

## Validation

Validated against the local images on 2026-09-17:

- 800 readable image files: exactly 100 in each metric.
- 800 distinct filenames, file contents, and decoded RGB images.
- 800 distinct coordinate pairs at six decimal places.
- 600 recorded Street View headings; 542 recorded panorama IDs.
- No missing image records, duplicate records, or out-of-range coordinates/headings.
- The 58 `extra_images/` replacements were excluded from all counts above.
