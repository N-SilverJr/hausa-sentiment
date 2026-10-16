"""
Hausa tweet sentiment â€“ Gradio demo.

Models:
  1. Classical (main): word + character TF-IDF + calibrated LinearSVC (test macro-F1 0.782)
  2. Recurrent: BiLSTM + attention pooling, Word2Vec mixed (test macro-F1 0.746)
  3. Transformer (optional): AfroXLMR-mini LoRA T1 (test macro-F1 0.590) when files + libs exist
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
# Space layout: app.py at repo root, models/ next to it.
# Local package layout: app/app.py with models/ one level up.
if (HERE / "models").exists():
    ROOT = HERE
elif (HERE.parent / "models").exists():
    ROOT = HERE.parent
else:
    ROOT = HERE
MODELS = ROOT / "models"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSICAL_PATH = MODELS / "classical_best.pkl"
RNN_PATH = MODELS / "rnn_R7_BiLSTM_attention_pooling.pt"
TRANSFORMER_DIR = MODELS / "transformer_T1_afro-xlmr-mini_LoRA_(CPU)"

LABELS = ["positive", "neutral", "negative"]  # ids 0, 1, 2

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = CTRL_RE.sub(" ", t)
    t = URL_RE.sub("<url>", t)
    t = MENTION_RE.sub("@user", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_bow(text: str) -> str:
    t = clean_text(text)
    t = t.replace("@user", " ").replace("<url>", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def simple_tokenize(text: str) -> list[str]:
    """Whitespace + light punctuation split used for the RNN vocabulary."""
    t = clean_text(text).lower()
    t = re.sub(r"([.,!?;:\"()\[\]{}])", r" \1 ", t)
    return [tok for tok in t.split() if tok]


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
    tokens = re.findall(r"[a-zA-ZÉ“É—Æ™Æ´ÆÆŠÆ˜Æ³']+", text.lower())
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
# Classical
# ---------------------------------------------------------------------------
_classical = None


def load_classical():
    global _classical
    if _classical is None:
        if not CLASSICAL_PATH.exists():
            raise FileNotFoundError(f"Classical model not found at {CLASSICAL_PATH}")
        import classical_model  # noqa: F401
        _classical = joblib.load(CLASSICAL_PATH)
    return _classical


def classical_predict_proba(bow: str) -> np.ndarray:
    bundle = load_classical()
    X = bundle["vectorizer"].transform([bow])
    clf = bundle["clf"]
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[0]
    scores = clf.decision_function(X)[0]
    exp = np.exp(scores - np.max(scores))
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# Recurrent (BiLSTM + attention)
# ---------------------------------------------------------------------------
_rnn = None
_rnn_word2id = None
_rnn_max_len = 64


def _build_rnn_module(config: dict, vocab_size: int):
    import torch
    import torch.nn as nn

    emb_dim = int(config["emb_dim"])
    hidden = int(config["hidden"])
    num_layers = int(config.get("num_layers", 1))
    dropout = float(config.get("dropout", 0.3))

    class BiLSTMAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.encoder = nn.LSTM(
                emb_dim,
                hidden,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # state_dict: attn.0 (256->64), attn.2 (64->1)  â†’ Sequential Linear, Tanh, Linear
            self.attn = nn.Sequential(
                nn.Linear(hidden * 2, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.classifier = nn.Linear(hidden * 2, 3)

        def forward(self, input_ids):
            # input_ids: (B, T)
            mask = (input_ids != 0).float()  # (B, T)
            emb = self.embedding(input_ids)
            out, _ = self.encoder(emb)  # (B, T, 2H)
            scores = self.attn(out).squeeze(-1)  # (B, T)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
            pooled = (out * weights).sum(dim=1)  # (B, 2H)
            return self.classifier(pooled)

    return BiLSTMAttention()


def load_rnn():
    global _rnn, _rnn_word2id, _rnn_max_len
    if _rnn is not None:
        return _rnn
    import torch

    if not RNN_PATH.exists():
        raise FileNotFoundError(f"RNN checkpoint not found at {RNN_PATH}")
    ckpt = torch.load(RNN_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    word2id = ckpt["word2id"]
    model = _build_rnn_module(config, vocab_size=len(word2id))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _rnn = model
    _rnn_word2id = word2id
    _rnn_max_len = int(ckpt.get("max_len", config.get("max_len", 64)))
    return _rnn


def rnn_predict_proba(text: str) -> np.ndarray:
    import torch

    model = load_rnn()
    unk = _rnn_word2id.get("<unk>", 1)
    ids = [_rnn_word2id.get(tok, unk) for tok in simple_tokenize(text)]
    if not ids:
        ids = [unk]
    ids = ids[:_rnn_max_len]
    if len(ids) < _rnn_max_len:
        ids = ids + [0] * (_rnn_max_len - len(ids))
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Transformer (optional)
# ---------------------------------------------------------------------------
_tokenizer = None
_transformer = None
_transformer_ok = False


def try_load_transformer() -> bool:
    global _tokenizer, _transformer, _transformer_ok
    if _transformer_ok or _transformer is not None:
        return _transformer_ok
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel

        base = "Davlan/afro-xlmr-mini"
        _tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
        base_model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3)
        _transformer = PeftModel.from_pretrained(base_model, str(TRANSFORMER_DIR))
        _transformer.eval()
        _transformer_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"[app] Transformer not loaded ({exc}). Classical + RNN only.")
        _transformer_ok = False
    return _transformer_ok


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(text: str):
    empty = {}
    if not text or not text.strip():
        return "Type or paste a Hausa tweet.", empty, empty, empty

    bow = text_bow(text)
    if not bow:
        return "Nothing left after cleaning (only links, mentions or symbols).", empty, empty, empty

    note = domain_note(text)
    prepared = f"Text seen after cleaning: {bow}"
    if note:
        prepared += f"\n\nâš  {note}"

    # Classical
    c_probs = classical_predict_proba(bow)
    classical_out = {lab: float(p) for lab, p in zip(LABELS, c_probs)}
    c_lab = LABELS[int(np.argmax(c_probs))]

    # RNN
    rnn_out = {}
    try:
        r_probs = rnn_predict_proba(text)
        rnn_out = {lab: float(p) for lab, p in zip(LABELS, r_probs)}
        r_lab = LABELS[int(np.argmax(r_probs))]
    except Exception as exc:  # noqa: BLE001
        prepared += f"\n(RNN not available: {exc})"
        r_lab = None

    # Transformer
    transformer_out = {}
    t_lab = None
    if try_load_transformer():
        import torch

        clean = clean_text(text)
        enc = _tokenizer(clean, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = _transformer(**enc).logits[0]
            tprobs = torch.softmax(logits, dim=-1).cpu().numpy()
        transformer_out = {lab: float(p) for lab, p in zip(LABELS, tprobs)}
        t_lab = LABELS[int(np.argmax(tprobs))]
    else:
        prepared += "\n(Transformer second opinion not available in this environment.)"

    # Agreement note
    labels_seen = [x for x in (c_lab, r_lab, t_lab) if x is not None]
    if len(set(labels_seen)) == 1 and len(labels_seen) > 1:
        prepared += "\nAll loaded models agree."
    elif len(set(labels_seen)) > 1:
        prepared += "\nModels disagree â€” treat the prediction as uncertain."

    return prepared, classical_out, rnn_out, transformer_out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
# Hausa tweet sentiment

Classifies Hausa social-media text as **positive**, **neutral** or **negative**.

- **Main model:** word + character TF-IDF + calibrated LinearSVC (test macro-F1 **0.782**)
- **Recurrent:** BiLSTM + attention pooling (test macro-F1 **0.746**)
- **Transformer:** AfroXLMR-mini LoRA (test macro-F1 **0.590**) when files and libraries are present
- Emoji and Hausa hooked letters are kept; links and @mentions are normalised
- Sarcasm and subtle negation are not handled reliably
- Trained on Hausa tweets â€” English or formal news prose may mislead the models
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
        placeholder="Rubuta ko manna tweet na Hausa a nanâ€¦",
    )
    btn = gr.Button("Classify", variant="primary")
    info = gr.Textbox(label="Prepared text and notes", interactive=False)
    with gr.Row():
        classical_label = gr.Label(label="Classical model (main) â€” 0.782")
        rnn_label = gr.Label(label="BiLSTM + attention â€” 0.746")
        transformer_label = gr.Label(label="Transformer LoRA â€” 0.590")
    btn.click(
        predict,
        inputs=box,
        outputs=[info, classical_label, rnn_label, transformer_label],
    )
    gr.Examples(EXAMPLES, inputs=box)

if __name__ == "__main__":
    import gradio.networking as _networking
    # Render does not expose localhost the way Gradio expects
    _networking.url_ok = lambda *args, **kwargs: True

    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )

