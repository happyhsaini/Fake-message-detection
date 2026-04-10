import os

# Always resolve relative to project root (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH       = os.path.join(BASE_DIR, "combined_data.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model_info.pkl")

RANDOM_STATE = 42
TEST_SIZE    = 0.2
MAX_FEATURES = 10000

# ── Prediction thresholds ──────────────────────────────────────────────────
# fake_prob < 40%       → Real     ✅
# fake_prob 40% - 60%   → Suspicious ⚠
# fake_prob > 60%       → Fake    🚫
FAKE_THRESHOLD   = 0.60
SUSPICIOUS_LOW   = 0.40
SUSPICIOUS_HIGH  = 0.60

# ── Trusted domain patterns (reduces false positives) ─────────────────────
TRUSTED_DOMAINS = [
    "github.com", "google.com", "microsoft.com", "apple.com",
    "amazon.com", "linkedin.com", "twitter.com", "facebook.com",
    "leetcode.com", "stackoverflow.com", "netflix.com", "spotify.com",
    "paypal.com", "stripe.com", "youtube.com", "zoom.us",
]
