# Sentiment Classification for Hausa Tweets

Project links: GitHub repository: [to be added] · Live system (Hugging Face Space): [to be added] · Demo video: [to be added] · Models: [to be added]

## 1. Abstract

A word-plus-character TF-IDF model with a calibrated LinearSVC reaches test macro-F1 **0.782** on the Hausa portion of AfriSenti, outperforming the best recurrent model (BiLSTM with attention, 0.746) and a budget-constrained LoRA fine-tune of AfroXLMR-mini (0.590). The classical model is therefore the primary system deployed in the public Gradio app.

This study classifies Hausa social-media text as positive, neutral or negative. The data come from AfriSenti-SemEval 2023 Subtask A (train 14 172, development 2 677, test 5 303). Two data-integrity problems affect evaluation: 215 development examples are exact duplicates of training rows, and the released test split has a markedly different surface form from train/development (almost no emoji, different cleaning of mentions and links). Model selection therefore used a de-leaked development split (`dev_noleak`, 2 462 rows). Character n-grams prove especially useful for Hausa tweets, which show spelling variation, missing diacritics and code-switching. Error analysis (from the original pipeline) shows neutral as the most confused class and confirms that confidence alone is not a reliable indicator of out-of-domain input. The classical model, the best recurrent checkpoint and the LoRA adapter are packaged for a Gradio web application.

## 2. Introduction

Hausa is spoken by tens of millions of people across West Africa, yet labelled resources for social-media NLP remain limited. Sentiment analysis of tweets and similar short texts is a practical application: it lets researchers and organisations track public opinion in a language that mainstream tools handle poorly.

AfriSenti (Muhammad et al., 2023) supplies labelled tweets for several African languages, including Hausa. This project uses the Hausa portion to answer four concrete questions:

1. How well can a simple classical model (TF-IDF + linear classifier) perform?
2. Do recurrent models that use Word2Vec embeddings improve on the classical baseline?
3. Does a pretrained African-language transformer improve further when fine-tuned under a realistic compute budget?
4. How sensitive are the models to the surface-form mismatch between the training data and the released test set, and to out-of-domain input?

The work covers the full lifecycle from data verification through training, evaluation, error analysis and deployment, but the present clean package concentrates on the results that matter for the report and the live system.

## 3. Data

### 3.1 Source and sizes

| Split        | Rows  | Notes                                      |
|--------------|------:|--------------------------------------------|
| Train        | 14 172 | Official AfriSenti Hausa training set     |
| Development  |  2 677 | Official development set                  |
| Dev (no leak)|  2 462 | Development after removing 215 train duplicates |
| Test         |  5 303 | Official test set                         |

Labels are encoded as `0 = positive`, `1 = neutral`, `2 = negative`. The mapping was verified against both the Hugging Face mirror and the official GitHub release; the two sources agree row by row.

### 3.2 Integrity findings

- **Leakage.** 215 development examples appear verbatim in the training split. All model selection reported in this package uses the de-leaked `dev_noleak` split.
- **Surface mismatch.** Train and development are almost entirely emoji-bearing. The released test set contains almost no emoji and was cleaned differently (mentions, links, punctuation). Applying test-style cleaning to development text produces a performance drop similar to the train–test gap observed for some models.
- **Conflicting labels.** Thirteen texts appear with more than one label across the corpus.
- **Ordering.** The official files are ordered in contiguous label blocks rather than randomly shuffled.

These issues are the reason the original pipeline distinguishes `dev` from `dev_noleak` and runs explicit domain-shift probes.

### 3.3 Preprocessing

Text is normalised for Unicode, control characters are removed, URLs become `<url>`, user mentions become `@user`, and whitespace is collapsed. Hausa hooked letters (`ɓ`, `ɗ`, `ƙ`, `ƴ`) and emoji are preserved. For the classical bag-of-words models the placeholders `@user` and `<url>` are stripped, producing the `text_bow` column used by the strongest baseline.

## 4. Models

### 4.1 Classical baselines

Feature sets examined:

- word n-grams (TF-IDF),
- character n-grams (TF-IDF),
- the union of word and character n-grams,
- raw counts with Complement Naive Bayes.

Classifiers: Logistic Regression, calibrated LinearSVC, ComplementNB. The strongest configuration is **word + character TF-IDF with a sigmoid-calibrated LinearSVC** (63 031 features).

### 4.2 Recurrent models

Bidirectional LSTM, GRU and vanilla RNN architectures were trained on top of Word2Vec embeddings (100-dimensional) learned from an in-domain corpus and from a mixed corpus that also includes MasakhaNEWS and Hausa Wikipedia text. Embeddings were either frozen or fine-tuned; pooling variants included mean, max and additive attention. The best run is a **BiLSTM with attention pooling** on fine-tuned mixed embeddings (R7).

### 4.3 Transformer

A single budget-constrained run fine-tunes `Davlan/afro-xlmr-mini` with LoRA (rank 8, alpha 16, targeting query and value projections) for one epoch on CPU. Approximately 0.25 % of parameters are trainable. Longer GPU fine-tunes of larger African-language models were left for a separate Colab notebook and are not part of the numbers reported here.

## 5. Results

### 5.1 Headline comparison (official test split)

| Family               | Best run                              | Test macro-F1 |
|----------------------|---------------------------------------|--------------:|
| Majority baseline    | B0                                    |         0.168 |
| Classical            | B4 word+char TF-IDF + LinearSVC       |         0.782 |
| Classical (char only)| B3 char TF-IDF + LogReg               |         0.775 |
| Classical (word only)| B1 word TF-IDF + LogReg               |         0.739 |
| Recurrent            | R7 BiLSTM + attention                 |         0.746 |
| Transformer (LoRA)   | T1 AfroXLMR-mini, 1 epoch CPU         |         0.590 |

All figures are taken from the experiment registry (`results/experiments.json`) and were produced by the original training scripts.

### 5.2 Classical detail

Character n-grams alone already reach 0.775, showing that sub-word features capture a large part of the signal in Hausa social-media text (spelling variation, dropped diacritics, code-switching). Adding word n-grams on top yields a further small gain to 0.782. Logistic Regression and LinearSVC are nearly tied; the calibrated LinearSVC was retained for deployment because it supplies well-behaved probability estimates.

### 5.3 Recurrent detail

Attention pooling improves over mean or max pooling. Fine-tuning the Word2Vec embeddings helps. None of the recurrent configurations surpasses the classical word+character baseline.

### 5.4 Transformer detail

The CPU LoRA run reaches only 0.590 macro-F1. This is expected under a one-epoch, CPU-only budget and should not be read as evidence that transformers are inherently weaker for Hausa. A fair comparison would require multi-epoch GPU fine-tuning of a larger model (AfroXLMR-base, AfriBERTa, etc.).

### 5.5 Domain-shift probes

When the same classical model is evaluated on development text that has been cleaned to resemble the released test format, macro-F1 drops by a few points, consistent with the surface mismatch noted in Section 3.2. The effect is visible but smaller than the gap between classical and under-trained transformer models.

## 6. Error analysis and robustness (summary of original findings)

- Neutral is the class most often confused with the other two.
- Code-switched, emoji-heavy and very short tweets remain error-prone.
- High-confidence predictions on non-Hausa or formal-news text are common; confidence alone is not a reliable language or domain detector.
- Zero-shot transfer of the Hausa-trained transformer to other AfriSenti languages (Amharic, Igbo, Yoruba, Swahili, \ldots) is substantially weaker than in-language performance.

A dedicated error-analysis notebook will expand these points once the remaining prediction CSVs are added to the clean package.

## 7. Deployment

The public system is a Gradio application that loads:

1. the classical word+character LinearSVC model (primary),
2. optionally the best recurrent checkpoint or the LoRA adapter as a second opinion.

Preprocessing is identical to the training pipeline. A lightweight heuristic warns the user when the input appears to be English or otherwise outside the intended domain; the heuristic is not a full language-identification system. The application is intended for Hugging Face Spaces (or an equivalent free hosting service).

## 8. Limitations

- The transformer experiment is under-powered (one CPU epoch). Stronger transformer results would require additional compute.
- Label quality of the original AfriSenti Hausa annotations was not independently re-annotated.
- The test set’s surface distribution differs from real user input that still contains emoji and raw mentions; reported test scores therefore reflect a partially cleaned register.
- Out-of-domain confidence is poorly calibrated; the system can be confidently wrong on English or formal Hausa prose.
- Sarcasm and subtle negation remain difficult for all model families examined.

## 9. Conclusion

On the Hausa AfriSenti benchmark a carefully engineered classical model currently outperforms both recurrent networks and a budget-constrained transformer. Character n-grams are the single most useful feature family for this data. The classical model is therefore the system shipped in the public demo, while the recurrent and transformer artefacts are retained for comparison and future work.

## References

- Muhammad, S. H., Abdulmumin, I., Ayele, A. A., et al. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. In Proceedings of EMNLP 2023.
- Alabi, J. O., et al. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In Proceedings of COLING 2022.
- Ogueji, K., Zhu, Y., and Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resource languages.
- Joshi, P., et al. (2020). The state and fate of linguistic diversity and inclusion in the NLP world.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing.
- Abid, A., et al. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild.
- AfriSenti dataset: `shmuhammad/AfriSenti-twitter-sentiment` (Hugging Face, CC BY 4.0).
- Base model: `Davlan/afro-xlmr-mini` (Hugging Face Hub).
