# Hausa Sentiment Classification

Three-class sentiment analysis (positive / neutral / negative) for Hausa social-media text, using the Hausa portion of **AfriSenti-SemEval 2023 Subtask A**.

This is a clean, notebook-first package modelled on a lightweight project template. It re-uses trained models and the experiment registry from the original research pipeline rather than re-running every experiment from scratch.

**Headline result.** A classical word + character TF-IDF model with a calibrated LinearSVC reaches **test macro-F1 0.782**, outperforming the best recurrent model (0.746) and a budget-constrained LoRA transformer (0.590).

---

## Project links

| Resource | Link |
|----------|------|
| GitHub repository | *[to be added]* |
| Live demo (Hugging Face Space) | *[to be added]* |
| Demo video | *[to be added]* |
| Models | local `models/` folder (classical + RNN checkpoints + LoRA adapter) |

---

## Repository structure

```
hausa-sentiment/
├── README.md
├── classical_model.py                              # needed to unpickle classical_best.pkl
├── notebooks/
│   ├── sentiment-classification.ipynb              # main end-to-end notebook
│   └── sentiment-classification-error-analysis.ipynb
├── report/
│   └── Hausa_Sentiment_Report.md
├── app/
│   ├── app.py                                      # Gradio web app
│   ├── requirements.txt
│   └── README.md
├── data/processed/
│   ├── afrisenti_hau_train.csv                     # 14 172
│   ├── afrisenti_hau_dev.csv                       #  2 677
│   ├── afrisenti_hau_dev_noleak.csv                #  2 462 (leakage removed)
│   └── afrisenti_hau_test.csv                      #  5 303
├── models/
│   ├── classical_best.pkl                          # best classical model
│   ├── rnn_R7_BiLSTM_attention_pooling.pt          # best recurrent model
│   ├── rnn_R1 … R6 *.pt
│   ├── transformer_T1_afro-xlmr-mini_LoRA_(CPU)/   # main LoRA (test F1 0.590)
│   └── transformer_SMOKE_afro-xlmr-mini_lora/      # smoke test only (test F1 0.314)
└── results/
    └── experiments.json                            # full experiment registry
```

---

## Headline results (official test split)

| Family      | Best run                              | Test macro-F1 |
|-------------|---------------------------------------|--------------:|
| Majority    | B0 DummyClassifier                    |         0.168 |
| Classical   | B4 word+char TF-IDF + LinearSVC       |         0.782 |
| Classical   | B3 char TF-IDF + LogReg               |         0.775 |
| Classical   | B1 word TF-IDF + LogReg               |         0.739 |
| Recurrent   | R7 BiLSTM + attention pooling         |         0.746 |
| Transformer | T1 AfroXLMR-mini LoRA (1 epoch, CPU)  |         0.590 |

All numbers come from `results/experiments.json` and were produced by the original training scripts.

---

## Quick start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
# for the notebooks also:
pip install pandas matplotlib jupyter
```

### 2. Run the Gradio app

```bash
python app/app.py
```

Open the local URL (default `http://127.0.0.1:7860`).  
The classical model loads from `models/classical_best.pkl`. The transformer second opinion loads only if the adapter files and `transformers`/`peft` are available.

### 3. Explore the notebooks

```bash
jupyter notebook notebooks/
```

- `sentiment-classification.ipynb` — data loading, registry tables, model-family comparison, artefact checks.
- `sentiment-classification-error-analysis.ipynb` — qualitative error examples and a simple taxonomy on the training split.

### 4. Read the report

`report/Hausa_Sentiment_Report.md` — concise mini-paper covering problem, data, models, results, limitations and references.

---

## Problem, language and dataset (summary)

- **Task.** Three-class sentiment classification of Hausa tweets / social-media text.
- **Language.** Hausa (Afro-Asiatic, widely spoken in West Africa). Mainstream English-centric tools handle it poorly; AfriSenti provides a labelled benchmark.
- **Target users.** Researchers, NGOs and organisations that need to monitor public sentiment in Hausa without relying on English-only pipelines.
- **Dataset.** AfriSenti Hausa (Muhammad et al., 2023): train 14 172, development 2 677, test 5 303. Labels: positive / neutral / negative.
- **Key data issues.** 215 development rows leak from training; the released test set has almost no emoji and was cleaned differently from train/dev. Model selection used a de-leaked development split.

Full details are in the report and in the data-integrity notes inside the main notebook.

---

## Models (summary)

| Family | Approach | Why chosen |
|--------|----------|------------|
| Classical | Word + character TF-IDF → calibrated LinearSVC | Strong baseline for short noisy text; character n-grams handle spelling variation and missing diacritics |
| Recurrent | BiLSTM / GRU / RNN + Word2Vec (in-domain & mixed) + attention | Test whether sequence models beat bag-of-n-grams |
| Transformer | AfroXLMR-mini + LoRA (1 epoch, CPU) | African-language-adapted encoder under a realistic submission budget |

The classical model is the system shipped in the web app.

---

## Reproducibility notes

- Random seed used throughout the original pipeline: **42**.
- Experiment registry: `results/experiments.json` (single source of truth for reported scores).
- Classical inference does not require GPU or internet.
- Transformer second opinion requires `torch`, `transformers`, `peft` and will download the base model `Davlan/afro-xlmr-mini` on first use if not cached.
- All train / dev / dev_noleak / test CSVs are included under `data/processed/`.

---

## Limitations (short)

- Transformer run is under-powered (one CPU epoch).
- Test surface form differs from real emoji-bearing user input.
- Out-of-domain confidence is poorly calibrated.
- Sarcasm and subtle negation remain difficult.

---

## Licence and data

- Code in this package: use under the same terms as the original project repository.
- AfriSenti data: CC BY 4.0 (see Muhammad et al., 2023).
- Base transformer: `Davlan/afro-xlmr-mini` (Hugging Face Hub).
