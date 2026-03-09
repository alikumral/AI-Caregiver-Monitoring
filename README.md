# RAGOS - Caregiver Monitoring and Evaluation System

Graduation project for ENS 492.
This repository contains an end-to-end prototype that analyzes caregiver-child interactions from audio only (no camera feed), then generates structured insights and alerts with ML/LLM agents.

## Project Scope

RAGOS aims to:

- transcribe speech from caregiver-dependent conversations,
- separate speakers and classify them (Man/Woman/Child),
- run sentiment, toxicity, sarcasm, and topic analysis,
- score caregiver performance,
- generate parent-facing notifications when needed.

Project report file in this repo:
- `ENS 492 - Final Report #104.docx` (dated 04.05.2025)

## Repository Structure

```text
.
|- Speech2Text/
|  |- mp3_new.py             # Deep learning based STT + speaker classification
|  |- mp3_optimized.py       # Acoustic feature based faster baseline
|  |- compare.py             # Compares acoustic vs deep-learning pipelines
|  |- mp3_with_train.py      # Training-oriented speaker classification experiments
|  |- interactive_labeler.py # GUI tool for manual speaker labeling
|  |- RTS/                   # Real-time chunk transcription/diarization experiments
|  `- comparison_results/    # Generated comparison reports and visualizations
|
|- LLM Evaluation/
|  |- app.py                 # Streamlit dashboard
|  |- backend/main.py        # FastAPI service layer
|  |- agents/
|  |  |- analysis/           # HF-based lightweight analysis agents
|  |  |- llm/                # Ollama-based generation/judgment agents
|  |  |- orchestration/      # Multi-agent execution pipeline
|  |  `- test/               # Evaluation and judge helpers
|  |- extract_text.py        # PDF -> Chroma embedding index
|  |- data/                  # Guidance documents and test outputs
|  `- firebase/              # Firebase initialization
|
`- ENS 492 - Final Report #104.docx
```

## High-Level Architecture

1. Speech processing layer
- `faster-whisper` for transcription
- `pyannote/speaker-diarization-3.1` for diarization
- `wav2vec2` variants for speaker demographic classification

2. Multi-agent LLM/NLP layer
- Sentiment: `cardiffnlp/twitter-roberta-base-sentiment`
- Toxicity: `unitary/toxic-bert`
- Sarcasm: `cardiffnlp/twitter-roberta-base-irony`
- Categorization: `facebook/bart-large-mnli`
- Caregiver scoring and notification generation via Ollama models

3. Application layer
- Streamlit operator dashboard (`LLM Evaluation/app.py`)
- FastAPI backend (`LLM Evaluation/backend/main.py`)
- Firebase/Firestore timeline, persistence, and notifications

## Key Findings (from the final report)

- End-to-end latency was reduced from minutes to near real-time behavior.
- Deep-learning speaker classification achieved higher quality than acoustic heuristics.
- In report benchmarks, deep-learning approach is slower (~1.24x) but more accurate/confident.
- A 125-day caregiver score series was forecasted for 6 months using Exponential Smoothing.

## Setup

### 1) Prerequisites

- Python 3.10+ (3.11 recommended)
- `ffmpeg`
- Optional CUDA-capable GPU
- Ollama (for local LLM agents)
- Firebase service account (for backend notification/storage flow)

### 2) LLM Evaluation environment

```powershell
cd "LLM Evaluation"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r backend\requirements.txt
```

Additional packages may be needed since code usage is broader than current requirements files:

```powershell
pip install streamlit streamlit-lottie evidently nest_asyncio plotly transformers torch chromadb langdetect argostranslate langchain-chroma google-cloud-firestore python-dateutil
```

### 3) Speech2Text environment

This folder currently has no canonical `requirements.txt`. Install based on scripts:

```powershell
cd "..\Speech2Text"
python -m venv .venv
.venv\Scripts\activate
pip install torch torchaudio transformers faster-whisper pyannote.audio librosa soundfile numpy pandas matplotlib scikit-learn tqdm xgboost joblib pygame pyaudio webrtcvad resemblyzer speechbrain seaborn
```

## Environment Variables

Create `.env` under `LLM Evaluation`:

```env
RAGOS_API_KEY=your_api_key
COLLECTION_NAME=analysis_results
FIREBASE_CREDENTIALS=path/to/serviceAccount.json
FIREBASE_BUCKET=your_project.appspot.com
GOOGLE_APPLICATION_CREDENTIALS=path/to/serviceAccount.json
HF_API_KEY=your_huggingface_token
```

Pull local Ollama models:

```powershell
ollama pull openhermes:7b-mistral-v2.5-q5_1
ollama pull qwen:7b
```

## Running the Project

### A) Streamlit dashboard

```powershell
cd "LLM Evaluation"
streamlit run app.py
```

### B) FastAPI backend

```powershell
cd "LLM Evaluation"
uvicorn backend.main:app --reload --port 8000
```

Important endpoints:

- `GET /health`
- `POST /analyze`
- `GET /aggregate/{user_id}`
- `GET /timeline/{user_id}`

### C) Speech2Text examples

Deep learning pipeline:

```powershell
cd "Speech2Text"
python mp3_new.py mp3s\a.mp3 -o transcript.txt --model small
```

Acoustic baseline:

```powershell
python mp3_optimized.py mp3s\a.mp3 -o transcript.txt --model small
```

Compare both:

```powershell
python compare.py --audio_dir mp3s --output_dir comparison_results
```

Manual speaker labeling GUI:

```powershell
python interactive_labeler.py --samples speaker_samples --labels speaker_labels.json
```

## Example Outputs in Repo

- Transcript sample: `Speech2Text/transcript.txt`
- Detailed speaker report: `Speech2Text/transcript_detailed_report.txt`
- Comparison visuals: `Speech2Text/comparison_results/visualization_results/`
- LLM test outputs: `LLM Evaluation/data/tests/`

## Security and Publishing Notes

- This repo contains experimental scripts and generated artifacts.
- Hardcoded Hugging Face tokens were replaced with `HF_API_KEY` environment usage in Speech2Text scripts.
- Consider adding Git LFS for large media (`.mp3`, `.wav`) before public release.
- `LLM Evaluation/README.md` is empty; this root README is the primary documentation.

## Ethics and Privacy

The project follows an audio-first monitoring approach intended to be more privacy-preserving than camera-based surveillance.
Outputs are decision-support signals and should not replace human review.

