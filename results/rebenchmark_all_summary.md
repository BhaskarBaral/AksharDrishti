# Fresh re-benchmark: all 4 engines, all splits, real vs synthetic

| Script | Engine | Real CER (n) | Synthetic CER (n) |
|---|---|---|---|
| bengali | easyocr | 0.8475 (15) | 0.1147 (90) |
| bengali | paddleocr-finetuned | 0.1623 (15) | 0.6358 (90) |
| bengali | tesseract | 0.8776 (15) | 0.0729 (90) |
| devanagari | easyocr | 0.7719 (15) | 0.1421 (90) |
| devanagari | paddleocr | 0.5652 (15) | 0.0472 (90) |
| devanagari | paddleocr-finetuned | 0.1561 (15) | 0.5766 (90) |
| devanagari | tesseract | 1.0563 (15) | 0.0614 (90) |
| devanagari | trocr | 1.0 (15) | 1.0 (90) |
| gujarati | paddleocr-finetuned | 0.3318 (15) | - |
| gujarati | tesseract | 0.8089 (15) | - |
| kannada | easyocr | 0.7208 (15) | - |
| kannada | paddleocr-finetuned | 0.0865 (15) | - |
| kannada | tesseract | 0.7681 (15) | - |
| malayalam | paddleocr-finetuned | 0.0674 (15) | - |
| malayalam | tesseract | 0.9023 (15) | - |
| odia | paddleocr-finetuned | 0.2393 (15) | - |
| odia | tesseract | 0.9917 (15) | - |
| punjabi | paddleocr-finetuned | 0.2241 (15) | - |
| punjabi | tesseract | 0.9757 (15) | - |
| tamil | easyocr | 1.0 (15) | 1.0 (90) |
| tamil | paddleocr | 0.8065 (15) | 0.0262 (90) |
| tamil | paddleocr-finetuned | 0.035 (15) | 0.3701 (90) |
| tamil | tesseract | 0.9831 (15) | 0.1009 (90) |
| telugu | easyocr | 0.9196 (15) | - |
| telugu | paddleocr-finetuned | 0.0973 (15) | - |
| telugu | tesseract | 0.9381 (15) | - |
| urdu | easyocr | 0.6106 (15) | 0.656 (90) |
| urdu | paddleocr-finetuned | 0.1284 (15) | 0.884 (90) |
| urdu | tesseract | 0.9011 (15) | 0.5811 (90) |

## Overall, REAL fields only (all scripts each engine supports)
| Engine | Mean CER | N fields | Scripts covered |
|---|---|---|---|
| easyocr | 0.8117 | 90 | bengali, devanagari, kannada, tamil, telugu, urdu |
| paddleocr | 0.6859 | 30 | devanagari, tamil |
| paddleocr-finetuned | 0.1528 | 150 | bengali, devanagari, gujarati, kannada, malayalam, odia, punjabi, tamil, telugu, urdu |
| tesseract | 0.9203 | 150 | bengali, devanagari, gujarati, kannada, malayalam, odia, punjabi, tamil, telugu, urdu |
| trocr | 1.0 | 15 | devanagari |

## Head-to-head on contested scripts ['bengali', 'devanagari', 'gujarati', 'kannada', 'malayalam', 'odia', 'punjabi', 'tamil', 'telugu', 'urdu'] (real fields only)
| Script | Engine | Mean CER | N |
|---|---|---|---|
| bengali | easyocr | 0.8475 | 15 |
| bengali | paddleocr-finetuned | 0.1623 | 15 |
| bengali | tesseract | 0.8776 | 15 |
| devanagari | easyocr | 0.7719 | 15 |
| devanagari | paddleocr | 0.5652 | 15 |
| devanagari | paddleocr-finetuned | 0.1561 | 15 |
| devanagari | tesseract | 1.0563 | 15 |
| devanagari | trocr | 1.0 | 15 |
| gujarati | paddleocr-finetuned | 0.3318 | 15 |
| gujarati | tesseract | 0.8089 | 15 |
| kannada | easyocr | 0.7208 | 15 |
| kannada | paddleocr-finetuned | 0.0865 | 15 |
| kannada | tesseract | 0.7681 | 15 |
| malayalam | paddleocr-finetuned | 0.0674 | 15 |
| malayalam | tesseract | 0.9023 | 15 |
| odia | paddleocr-finetuned | 0.2393 | 15 |
| odia | tesseract | 0.9917 | 15 |
| punjabi | paddleocr-finetuned | 0.2241 | 15 |
| punjabi | tesseract | 0.9757 | 15 |
| tamil | easyocr | 1.0 | 15 |
| tamil | paddleocr | 0.8065 | 15 |
| tamil | paddleocr-finetuned | 0.035 | 15 |
| tamil | tesseract | 0.9831 | 15 |
| telugu | easyocr | 0.9196 | 15 |
| telugu | paddleocr-finetuned | 0.0973 | 15 |
| telugu | tesseract | 0.9381 | 15 |
| urdu | easyocr | 0.6106 | 15 |
| urdu | paddleocr-finetuned | 0.1284 | 15 |
| urdu | tesseract | 0.9011 | 15 |
