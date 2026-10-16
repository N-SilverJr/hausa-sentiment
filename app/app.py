"""
Hausa tweet sentiment – Gradio demo.

Primary model: word + character TF-IDF + calibrated LinearSVC (test macro-F1 0.782).
Optional second opinion: AfroXLMR-mini LoRA adapter (if the files and libraries are available).

Preprocessing mirrors the original pipeline:
  Unicode NFC → strip control chars → URL → <url> → @mention → @user → collapse whitespace.
For the classical bag-of-words model the placeholders @user and <url> are then removed.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Paths – work both when the app is run from the repo root and from app/
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "models").exists() else HERE
MODELS = ROOT / "models"

# Allow unpickling of the custom UnionVectorizer saved with the classical model
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLASSICAL_PATH = MODELS / "classical_best.pkl"
TRANSFORMER_DIR = MODELS / "transformer_T1_afro-xlmr-mini_LoRA_(CPU)"

LABELS = ["positive", "neutral", "negative"]  # ids 0, 1, 2

# ---------------------------------------------------------------------------
# Preprocessing (same rules as the original project)
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_text(text: str) -> str:
    """Full cleaning used by neural / transformer models."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = CTRL_RE.sub(" ", t)
    t = URL_RE.sub("<url>", t)
    t = MENTION_RE.sub("@user", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_bow(text: str) -> str:
    """Classical bag-of-words view: placeholders stripped."""
    t = clean_text(text)
    t = t.replace("@user", " ").replace("<url>", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Lightweight domain warning (not a full language ID system)
# ---------------------------------------------------------------------------
HAUSA_MARKERS = {
    "da", "ne", "ce", "ba", "ko", "sai", "kuma", "amma", "wani", "wata",
    "nan", "can", "yana", "yake", "ake", "ana", "ina", "muna", "suna",
    "allah", "wallahi", "gaskiya", "to", "kai", "shi", "ita", "su", "mu",
    "ku", "ni", "ka", "ki", "ya", "ta", "a", "na", "mai", "masu",
}
ENGLISH_MARKERS = {
    "the", "and", "is", "are", "was", "were", "this", "that", "with",
    "for", "you", "have", "has", "not", "but", "from", "they", "will",
}


def domain_note(text: str) -> str:
    tokens = re.findall(r"[a-zA-ZɓɗƙƴƁƊƘƳ']+", text.lower())
    if len(tokens) < 2:
        return "Input is very short; the prediction may be unreliable."
    h = sum(1 for t in tokens if t in HAUSA_MARKERS)
    e = sum(1 for t in tokens if t in ENGLISH_MARKERS)
    if e >= 3 and h == 0:
        return "This looks more like English than Hausa. The model was trained on Hausa tweets."
    if h == 0 and len(tokens) > 6:
        return "Few common Hausa function words detected. Treat the prediction with caution."
    return ""


# ---------------------------------------------------------------------------
# Model loading (lazy, classical is required, transformer is optional)
# ---------------------------------------------------------------------------
_classical = None
_tokenizer = None
_transformer = None
_transformer_ok = False


def load_classical():
    """Load the classical model dict {vectorizer, clf, ...}."""
    global _classical
    if _classical is None:
        if not CLASSICAL_PATH.exists():
            raise FileNotFoundError(f"Classical model not found at {CLASSICAL_PATH}")
        import classical_model  # noqa: F401  – required for unpickling UnionVectorizer
        _classical = joblib.load(CLASSICAL_PATH)
    return _classical


def classical_predict_proba(text_bow: str):
    """Return probability vector [pos, neu, neg] for a single bow string."""
    bundle = load_classical()
    vec = bundle["vectorizer"]
    clf = bundle["clf"]
    X = vec.transform([text_bow])
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[0]
    scores = clf.decision_function(X)[0]
    exp = np.exp(scores - np.max(scores))
    return exp / exp.sum()


def try_load_transformer():
    """Attempt to load the LoRA adapter. Failures are non-fatal."""
    global _tokenizer, _transformer, _transformer_ok
    if _transformer_ok or _transformer is not None:
        return _transformer_ok
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel

        base = "Davlan/afro-xlmr-mini"
        _tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=3
        )
        _transformer = PeftModel.from_pretrained(base_model, str(TRANSFORMER_DIR))
        _transformer.eval()
        _transformer_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"[app] Transformer not loaded ({exc}). Classical model only.")
        _transformer_ok = False
    return _transformer_ok


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(text: str):
    if not text or not text.strip():
        return "Type or paste a Hausa tweet.", {}, {}

    bow = text_bow(text)
    if not bow:
        return "Nothing left after cleaning (only links, mentions or symbols).", {}, {}

    note = domain_note(text)
    prepared = f"Text the classical model saw: {bow}"
    if note:
        prepared = prepared + "\n\n⚠ " + note

    # --- classical (always) ---
    probs = classical_predict_proba(bow)
    classical_out = {lab: float(p) for lab, p in zip(LABELS, probs)}

    # --- optional transformer ---
    transformer_out = {}
    if try_load_transformer():
        import torch
        clean = clean_text(text)
        enc = _tokenizer(
            clean, return_tensors="pt", truncation=True, max_length=64
        )
        with torch.no_grad():
            logits = _transformer(**enc).logits[0]
            tprobs = torch.softmax(logits, dim=-1).cpu().numpy()
        transformer_out = {lab: float(p) for lab, p in zip(LABELS, tprobs)}
        c_lab = LABELS[int(np.argmax(probs))]
        t_lab = LABELS[int(np.argmax(tprobs))]
        if c_lab == t_lab:
            prepared += "\nBoth models agree."
        else:
            prepared += "\nThe models disagree. Treat this prediction as uncertain."
    else:
        prepared += "\n(Transformer second opinion not available in this environment.)"

    return prepared, classical_out, transformer_out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
# Hausa tweet sentiment

Classifies Hausa social-media text as **positive**, **neutral** or **negative**.

- **Main model:** word + character TF-IDF + calibrated LinearSVC  
  (test macro-F1 **0.782** on the official AfriSenti Hausa test set).
- **Second opinion:** AfroXLMR-mini LoRA adapter (loaded when the files and libraries are present).
- Emoji and Hausa hooked letters are kept; links and @mentions are normalised.
- Sarcasm and subtle negation are not handled reliably.
- The model was trained on Hausa tweets; English or formal news prose may give misleading results.
"""

EXAMPLES = [
    "Allah ya kare mana kasarmu Nigeria da kare warsa daga sharrin yan siyasa ameen",
    "Gaskiya Buhari kaci Amanan mu da kasa wannan mutum ah wajen nan",
    "Sai haquri fa yan madrid daman kunce champion din ya muku yawa",
    "MashaAllah Allah ya Kara daukaka darajar wanan Film",
]


with gr.Blocks(title="Hausa Sentiment") as demo:
    gr.Markdown(DESCRIPTION)
    box = gr.Textbox(
        label="Hausa text",
        lines=3,
        placeholder="Rubuta ko manna tweet na Hausa a nan…",
    )
    btn = gr.Button("Classify", variant="primary")
    info = gr.Textbox(label="Prepared text and notes", interactive=False)
    with gr.Row():
        classical_label = gr.Label(label="Classical model (main)")
        transformer_label = gr.Label(label="Transformer (second opinion)")
    btn.click(predict, inputs=box, outputs=[info, classical_label, transformer_label])
    gr.Examples(EXAMPLES, inputs=box)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)
