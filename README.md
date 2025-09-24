# Reply Classification (SvaraAI mini-project)

## Project Structure
- `data/` – dataset (replies.csv)
- `models/` – trained models (baseline + transformer)
- `src/` – preprocessing, training, evaluation scripts
- `app.py` – FastAPI service
- `requirements.txt` – dependencies
- `Dockerfile` – optional container
- `answers.md` – reasoning answers

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
