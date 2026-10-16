# Hausa Sentiment – Gradio app

## What it does

Classifies Hausa social-media text as positive, neutral or negative.

- **Primary model:** word + character TF-IDF + calibrated LinearSVC  
  (test macro-F1 0.782 on the official AfriSenti Hausa test set).
- **Optional second opinion:** AfroXLMR-mini LoRA adapter  
  (loaded only when the adapter files and `transformers`/`peft` are available).

## How to run locally

```bash
cd hausa-sentiment
pip install -r app/requirements.txt
python app/app.py
```

The classical model is loaded from `models/classical_best.pkl`.  
The transformer adapter is loaded from `models/transformer_T1_afro-xlmr-mini_LoRA_(CPU)/` if present.

## Limitations

- Trained on Hausa tweets; English or formal news prose can produce high-confidence but wrong predictions.
- Sarcasm and subtle negation are not handled reliably.
- The domain warning is a simple heuristic, not a full language-identification system.
