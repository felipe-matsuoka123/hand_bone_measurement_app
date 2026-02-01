# Hand Bone Measurements App

Streamlit app to compute Z-scores and percentiles for hand bone measurements using LMS curves.

## Requirements

- Python 3.9+
- Conda or venv (recommended)

## Setup (Conda)

```bash
conda env create -f environment.yml
conda activate hand-bone-app
```

If you update `environment.yml`, re-run the commands above to recreate the env.

## Setup (pip/venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install streamlit pandas numpy scipy matplotlib
```

## Install package (for imports)

This project uses a local package (`src`). Install it once so `import src...` works:

```bash
pip install -e .
```

## Data

Place the CSV file at:

```
data/BoneMeasurements.csv
```

The file is ignored by git (see `.gitignore`).

## Run the app

From the project root:

```bash
streamlit run scripts/app.py
```

## Notes

- If you edit the CSV and don’t see changes, clear Streamlit cache or restart the app.
- The UI is configured via `.streamlit/config.toml`.
