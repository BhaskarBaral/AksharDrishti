# Test data — upload-and-try samples per script

5 real handwriting word-crop images per script (Devanagari, Tamil, Telugu,
Urdu, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi), for use with the
demo's "Try your own image" upload panel.

These are copied from the sibling dataset repo's `annotated/` folder — the
exact held-out set `training/prepare_datasets.py` excludes from the
fine-tuning pool, so every image here is genuinely unseen by the fine-tuned
models. Each script's folder has a `manifest.json` listing the ground truth
for each image, so you can eyeball whether an engine's OCR output is right
without having to look it up elsewhere:

```json
[{ "filename": "deva_real_0082.jpeg", "groundTruth": "मल्लोय", "sourceSplit": "test" }, ...]
```

Usage: open the app, go to "4. Try your own image", pick the matching
script, upload one of these files, and compare the result to its
`manifest.json` entry.
