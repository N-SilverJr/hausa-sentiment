# Deploy to Render (free)

## 1. Put the project on GitHub

In PowerShell, from the project folder:

```powershell
cd C:\Users\hp\OneDrive\Desktop\hausa-sentiment

git init
git add app/ classical_model.py models/classical_best.pkl models/rnn_R7_BiLSTM_attention_pooling.pt models/transformer_T1_afro-xlmr-mini_LoRA_(CPU)/ requirements.txt render.yaml runtime.txt README.md
git commit -m "Hausa sentiment Gradio app for Render"

# Create an empty repo on GitHub named hausa-sentiment, then:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hausa-sentiment.git
git push -u origin main
```

Do **not** commit huge unused files (R1–R6, SMOKE, full data CSVs) if the free repo gets heavy.

## 2. Create the Web Service on Render

1. Open https://dashboard.render.com
2. **New → Web Service**
3. Connect the GitHub repo
4. Settings:

| Field | Value |
|-------|--------|
| Name | hausa-sentiment |
| Region | closest to you |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python app/app.py` |
| Instance type | **Free** |

5. Click **Create Web Service**

## 3. Wait for the build

First build can take 10–20 minutes (PyTorch CPU wheel).  
When it succeeds, Render shows a URL like:

`https://hausa-sentiment.onrender.com`

## 4. Notes

- Free instances **spin down** after ~15 minutes idle; the first request after that can take 30–60 seconds.
- Transformer needs extra packages (`transformers`, `peft`); the free deploy ships **classical + RNN**. Add those packages later if the instance has enough memory.
- `PORT` is set by Render automatically; `app.py` already reads it.

## 5. Put the link in your README

```text
Live demo: https://hausa-sentiment.onrender.com
```
