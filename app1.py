"""
ShieldScan AI — Entry Point
Run from the project folder:
    python app1.py
"""
import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Check that model files exist before starting ───────────────────────────
MODEL_PATH      = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VECTORIZER_PATH):
    print("\n" + "="*60)
    print("  ❌  MODEL FILES NOT FOUND")
    print("="*60)
    print("\n  You need to train the model first.")
    print("  Run this command:\n")
    print("      python train_model.py\n")
    print("  Make sure combined_data.csv is in the same folder.")
    print("="*60 + "\n")
    sys.exit(1)

from src.app import app

if __name__ == "__main__":
    print("\n✅ Model loaded successfully.")
    print("🌐 Starting ShieldScan AI at http://127.0.0.1:5500\n")
    app.run(host="127.0.0.1", port=5500, debug=True)
