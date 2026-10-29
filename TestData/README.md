# TestData — the demo's held-out test set

Everything the demo's OCR and Layout tabs offer as samples comes from here.
Built by `_tools/build_testdata.py` (re-run it to rebuild; it copies
read-only from the dataset repo `Akshar_dristhii_code`).

```
TestData/
├── <language>/              devanagari bengali tamil telugu urdu gujarati
│   │                        kannada malayalam odia punjabi
│   ├── word_level/          real handwritten word crops + manifest.json
│   └── page_level/          handwritten pages + manifest.json
├── layout/                  layout detection, kept apart from the languages
│   ├── real_labelled/       20 real IndicDLP Odia test pages + region answer key
│   ├── real_unlabelled/     8 real complex forms (worksheets, bank forms)
│   └── synthetic_labelled/  39 synthetic forms (4 per language) with seals,
│                            stamps, signatures + region answer key
└── summary.json             counts per folder
```

## What the dashboard shows

| Tab | Picker | Reads |
|---|---|---|
| OCR → Word level | language pills → word buttons | `<language>/word_level/` |
| OCR → Page level | language pills → page buttons | `<language>/page_level/` |
| Layout | three labelled groups (synthetic narrowed to the picked language) | `layout/<group>/` |

## Held out from every model

* **Word crops** — none appear in the fine-tuned recognizers' training lists
  (`demo/training/datasets/<language>/train.txt`, `val.txt`). Human-reviewed
  held-out words (`humanReviewed: true`) are listed first.
* **Synthetic pages** — none had a field cropped into a synthetic-pilot
  training list; layout pages are only from the `test` hash bucket the layout
  fine-tune never saw.
* **IndicDLP Odia pages** — IndicDLP's own test split; the layout fine-tune
  trained on IndicDLP train shards only.

## Honest limits

| Language | word_level | page_level |
|---|---|---|
| devanagari | 50 real | 10 **synthetic** handwriting-style forms |
| bengali | 50 real | 10 **real** pages (ICDAR 2024 HWD) |
| telugu | 15 real | 10 **real** pages (ICDAR 2024 HWD) |
| tamil, urdu, gujarati, kannada, malayalam, odia, punjabi | 15 real | 10 **synthetic** handwriting-style forms |

* Only Devanagari and Bengali have unseen real words beyond the 15
  human-reviewed held-out ones; for the other languages the fine-tuning used
  every other real word on disk.
* Real handwritten *pages* exist only for Bengali and Telugu. The other eight
  languages' page_level folders hold synthetic forms rendered in
  handwriting-style fonts — every manifest entry and file name says
  `synthetic`.
* `real_unlabelled` pages have no answer key; judge them by eye.

## Manifest fields

`word_level/manifest.json`: `file`, `groundTruth`, `source` (`real`),
`humanReviewed`, `collectionSource`, `license`.
`page_level/manifest.json`: `file`, `groundTruth` (whole page), `source`
(`real` / `synthetic`); synthetic entries also carry `quality` and per-line
`lines` with ULCA `boundingBox`es.
`layout/*/manifest.json`: `file`, `script`, `source`, `regions`
(`[{class, bbox: [x0, y0, x1, y1]}]`, or `null` when unlabelled).
