# UrbanScore

UrbanScore uses OpenAI's CLIP model to understand apartment listing photos — brightness, luxury, and condition — and combines those visual signals with tabular listing data (price, area, location, year built) to power semantic image search and a "good deal" price model. The current dataset is scraped from [SUUMO](https://suumo.jp), a Japanese real estate portal, and the demo frontend is scoped to Tokyo listings.

## How it works

1. **Scrape** — `suumo_scraper.py` collects listing data and image URLs from SUUMO into `raw_data/`.
2. **Clean & embed** — `final_project_package/ml_logic/data_clean.py` cleans the tabular data; `final_project_package/embeddings/embeddings.py` generates CLIP image/text embeddings.
3. **Score** — `final_project_package/interface/score_images.py` scores each image (brightness / luxury / condition) using the Anthropic API.
4. **Train** — `final_project_package/ml_logic/model.py` and `preprocessor_pipeline.py` train a price model (scikit-learn) on the cleaned + embedded data to flag under/overvalued listings.
5. **Search & browse** — `frontend/app.py` is a Streamlit app: type a natural-language prompt (e.g. "kitchen with island"), filter by price/area/year, and browse results on an interactive map with "Good Deal" badges.

## Project structure

```
final_project_package/
  embeddings/        CLIP model loading, image/text embeddings, similarity
  ml_logic/           data cleaning, preprocessing pipeline, price model
  interface/          orchestration entry points (main_basic.py) + image scoring
frontend/
  app.py               Streamlit search & map UI
suumo_scraper.py        listing scraper
raw_data/                scraped listings & images
data_dump/               cleaned/embedded/scored CSVs (generated, not tracked)
assessment/               experiment notebooks & model evaluation
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs final_project_package (setup.py)
```

Create a `.env` file in the project root with:

```
ANTHROPIC_API_KEY=your-key-here
```

(Also needed if you use `gspread`/`google-auth` for Sheets integration — see `assessment/secrets`.)

## Running it

```bash
make run_load_data      # load raw scraped data
make run_embeddings     # generate CLIP embeddings for images
make run_preprocess      # clean + preprocess for modelling
make run_train           # train the price model
make run_evaluate        # evaluate the trained model
make run_pred            # generate predictions
make streamlit           # launch the Streamlit search/map UI
```

Image scoring (brightness/luxury/condition via Claude) is run separately:

```bash
python score_images.py          # pauses for confirmation after the first batch
python score_images.py --auto   # fully automated, for long unattended runs
```

## Known issues

- We maintain our own scraped dataset (no public listings API) — data quality varies by source.
- Price data is skewed with outliers.
- "Floating apartment" phenomenon — listings that reappear at different addresses/prices.
- CLIP embedding fails silently on corrupt images (handled with sentinel values in `score_images.py`).

## Team

Katia, Lance, and Bea.
