# ShieldScan AI — Fake Message Detection Platform

## Quick Start (VS Code)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image analysis)
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Ubuntu/WSL**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

### 3. Train the model (first time only)
```bash
python train_model.py
```
This reads `combined_data.csv` and creates `model.pkl`, `vectorizer.pkl`, `model_info.pkl`.

### 4. Run the website
```bash
python app1.py
```
Open: http://127.0.0.1:5500

---

## File Structure
```
project/
├── app1.py                  ← Entry point (run this)
├── train_model.py           ← Standalone training script
├── combined_data.csv        ← Training dataset
├── model.pkl                ← Trained model (auto-generated)
├── vectorizer.pkl           ← TF-IDF vectorizer (auto-generated)
├── model_info.pkl           ← Model metadata (auto-generated)
├── requirements.txt
├── src/
│   ├── app.py               ← All Flask routes
│   ├── config.py            ← Paths and thresholds
│   ├── preprocess.py        ← Text cleaning + CSV loading
│   ← feature_engineering.py ← TF-IDF builder
│   ├── explain.py           ← Human-readable explanations
│   ├── ocr.py               ← Image → text (Tesseract)
│   └── utils.py             ← pickle helpers
├── templates/
│   └── index.html           ← Full UI
└── static/
    ├── css/style.css
    └── js/script.js
```

## CSV Format for Training
| Column | Values |
|--------|--------|
| label / v1 | `ham` or `spam` (or `real`/`fake`) |
| text / v2 / message | message text |

## Prediction Logic
| Fake Probability | Label |
|-----------------|-------|
| < 40% | ✅ Real |
| 40–60% | ⚠ Suspicious |
| > 60% | 🚫 Fake |

Adjust thresholds in `src/config.py`.
