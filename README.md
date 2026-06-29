# Shape Pixel Extraction

This small utility detects the largest white shape on a black background and extracts:

- Per-column top/bottom y coordinates (vertical spans = height across columns).
- Per-row left/right x coordinates (horizontal spans = width across rows).

Usage:

```
python extract_shape_pixels.py path/to/image.png --out-dir results
```

Outputs written to `results/`:

- `vertical_spans.csv` — columns: `x, top_y, bottom_y, height`.
- `horizontal_spans.csv` — columns: `y, left_x, right_x, width`.
- `spans_visualization.png` — visualization with span endpoints marked.

Install dependencies:

```
pip install -r requirements.txt
```

Notes:

- Works best when the shape is white on a black background (or high contrast). Adjust `--thresh` if needed.
- For multiple shapes, the script selects the single largest contour by area. Modify `find_largest_contour` to change selection behavior.
