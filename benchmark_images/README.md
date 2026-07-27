# Benchmark Image Manifest

The image files are intentionally not distributed with VIDA-GEO. The
`benchmark_manifest.csv` file describes the expected benchmark without
containing image data or machine-specific absolute paths.

Expected layout:

```text
benchmark_images/
  safety/       100 GSV images, 512x512
  lively/       100 GSV images, 512x512
  beautiful/    100 GSV images, 512x512
  wealthy/      100 GSV images, 512x512
  boring/       100 GSV images, 512x512
  depressing/   100 GSV images, 512x512
  greenery/     100 satellite images, 1024x1024
  road_risk/    100 satellite images, 1024x1024
```

Manifest fields:

- `split`: benchmark split name.
- `modality`: `gsv` or `satellite`.
- `metric`: VIDA-GEO benchmark metric.
- `metric_index`: zero-based index within that metric.
- `relative_image_path`: expected path relative to the repository root.
- `latitude`, `longitude`: image-center coordinates.
- `width_px`, `height_px`: final input dimensions.
- `gsv_heading_degrees`: clockwise camera heading from true north; blank for
  satellite images.
- `heading_source`: whether the heading came from the GSV selection manifest or
  was recovered from the original legacy filename; `not_applicable` for
  satellite images.

The six GSV metrics contain 600 images total. The two satellite metrics contain
200 images total. Images under any local `extra_images/` directory are not part
of this 800-image benchmark.
