"""
Explain why a prediction was made by highlighting top influential words.
"""
import re


# Common spam / phishing trigger words (helps build human-readable reason)
SPAM_SIGNALS = [
    "free", "win", "winner", "prize", "claim", "click", "urgent", "limited",
    "offer", "congratulations", "selected", "lucky", "cash", "reward",
    "account", "verify", "password", "bank", "credit", "loan", "investment",
    "guaranteed", "risk free", "act now", "apply now", "dear customer",
    "you have been chosen", "otp", "kyc", "suspicious", "unauthorized",
]

HAM_SIGNALS = [
    "meeting", "schedule", "attached", "please find", "regards", "sincerely",
    "invoice", "receipt", "order", "delivery", "shipped", "confirmed",
    "update", "new features", "release", "digest", "newsletter",
    "unsubscribe", "preferences", "manage",
]


def explain_prediction(message: str, prediction: int, fake_prob: float) -> dict:
    """
    Return a dict with:
      - label   : "Fake" / "Real" / "Suspicious"
      - reason  : human-readable sentence
      - signals : list of trigger words found
      - tip     : advice for the user
    """
    msg_lower = message.lower()

    found_spam = [w for w in SPAM_SIGNALS if w in msg_lower]
    found_ham  = [w for w in HAM_SIGNALS  if w in msg_lower]

    if prediction == 1:
        label = "Fake / Spam"
        if found_spam:
            reason = (f"This message contains suspicious keywords: "
                      f"{', '.join(found_spam[:5])}.")
        else:
            reason = ("The language pattern closely matches spam/phishing "
                      "messages in the training data.")
        tip = ("Do not click any links. Do not share personal info. "
               "If from a known contact, verify by phone.")
    else:
        label = "Real / Legitimate"
        if found_ham:
            reason = (f"The message contains normal, expected phrases: "
                      f"{', '.join(found_ham[:5])}.")
        else:
            reason = "Language pattern matches typical legitimate messages."
        tip = "Looks safe, but always exercise caution with unknown senders."

    # Uncertain zone
    if 0.40 <= fake_prob <= 0.60:
        label  = "⚠ Suspicious — Needs Review"
        reason = (f"The model is not fully confident (fake probability: "
                  f"{fake_prob*100:.1f}%). Review manually before acting.")
        tip    = ("Do not click links. Verify sender identity before responding.")

    return {
        "label":   label,
        "reason":  reason,
        "signals": found_spam[:6] if prediction == 1 else found_ham[:4],
        "tip":     tip,
    }
