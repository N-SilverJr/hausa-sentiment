# Hausa Sentiment – Gradio app

## Models shown

| Panel | Model | Test macro-F1 |
|-------|--------|--------------:|
| Classical (main) | word + char TF-IDF + LinearSVC | 0.782 |
| BiLSTM + attention | R7 recurrent checkpoint | 0.746 |
| Transformer LoRA | AfroXLMR-mini T1 (optional) | 0.590 |

The transformer panel only fills when `transformers` + `peft` are installed and the adapter folder is present.

## Run locally

```bash
cd hausa-sentiment
pip install -r app/requirements.txt
# optional second opinion:
# pip install transformers peft
python app/app.py
```

Requires `classical_model.py` at the project root (for unpickling the classical model) and PyTorch for the RNN panel.

## Limitations

- Trained on Hausa tweets; English or formal news can look confident but wrong.
- Sarcasm and subtle negation are unreliable.
- Domain warning is a heuristic, not full language ID.
