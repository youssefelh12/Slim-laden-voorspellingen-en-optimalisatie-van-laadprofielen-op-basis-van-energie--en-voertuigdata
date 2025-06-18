# Hourly Energy Forecasting Project

> End‑to‑end pipeline for generating 24‑hour forecasts of **EV‑charger** usage. (Building‑demand models are included only for validation and benchmarking.)

---

## 📁 Repository layout

| Path                                            | What lives here                                                                                                                                                                                       |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Forecast_scripts/**                          | Scripts (and data) that produces a 24‑hour EV‑charger forecast for a given date                                                                                                               |
| **best_prediction_scripts_per_resolution/** | Curated “champion” models for each time‑resolution (monthly, daily, hourly) and for **buildings** & **chargers** separately. Swap models here to upgrade the pipeline without touching anything else. |
| **final_pipeline/**                            | Production orchestrator. Includes `call_api.py` for pulling raw data, `pipeline.py` for stitching everything together, and a Jupyter notebook that walks through the whole flow step‑by‑step.         |
| **archive/**                                    | The attic: every experiment, draft, and discarded approach is kept for posterity. Nothing here runs in production, but it’s a goldmine of ideas.                                                      |
| `.venv/`                                        | (Optional) local virtual environment.                                                                                                                                                                 |
| `config`                                        | Centralised defaults & helper functions (API keys, paths, feature flags…).                                                                                                                            |

---

## 🚀 Quick start

```bash
# 1. Clone and enter the repo
$ git clone [https://github.com/youssefelh12/Slim-laden-voorspellingen-en-optimalisatie-van-laadprofielen-op-basis-van-energie--en-voertuigdata.git]

# 2. Create & activate a virtual environment (recommended)
$ python -m venv .venv
$ source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# 3. Install the library and its dependencies
#    a) As an editable dev install:
$ pip install --upgrade pip
$ pip install -e .               # uses pyproject.toml / setup.cfg if present
#    b) From a consolidated requirements file (fallback):
$ pip install -r requirements.txt

# 4. Run a 24‑hour forecast for a given date
$ python Forecast_scripts/forecast_24h.py


> **Python ≥3.10** is strongly recommended. The project leans on **pandas**, **scikit‑learn**, **xgboost**, **lightgbm**, **statsmodels**; installation may take a minute on first‑time setups.

---

## 🏗️ Library installation options

Because the codebase evolves quickly—and has a fair few dependencies—you have two main ways to consume it:

### Editable install (development‑friendly)

```bash
pip install -e .
```

*Pros:* instant re‑load of local changes, perfect for notebook work.
*Cons:* ties your environment to your working copy.


## 📝 Usage patterns

### Generating a one‑off forecast

1. Choose the appropriate script in **Forecast_scripts/**
2. Edit the `TARGET_DATE` variable.
3. The script uses the historical data, trains a model, and outputs a CSV + prettified plot.

### Running end‑to‑end (API ➜ forecast ➜ artefacts)

```bash
python final_pipeline/pipeline.ipynb
```

* Pipeline pulls fresh data from api.
* Preprocesses & features engineers.
* Gives all the data to the model.
* Saves forecasts & evaluation metrics under `./results/`.

### Promoting a new model

1. Prototype inside **archive/**.
2. Once it beats the incumbent, move the script into **best\_prediction\_scripts\_per\_resolution/** under the appropriate sub‑folder.
3. Update the mapping in `final_pipeline/pipeline.py` (one import line).
4. Commit, open a PR, and merge once tests pass.

---

Made with ☕ and ⚡ by the Zeno Charged forecasting team.
