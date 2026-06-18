# Products

This repository is an AI-native project that builds a product recommendation system based on the [Amazon Reviews](https://amazon-reviews-2023.github.io/) dataset. I built it to demonstrate skill in building scalable, production-ready ML systems with real applications.

# Setup

Everything in this repository can be installed via running `poetry install` in the project root. Java is required to run Spark and build the pipelines. After installing the dependencies, run:

```
poetry run pipelines/feature-pipeline.py
poetry run pipelines/embed-products.py
poetry run pipelines/bm25-products.py
```

With the feature pipeline and product embeddings completed, the two tower model can be trained to learn long-term user preferences:

`poetry run training/train_two_tower.py`

Then only `poetry run streamlit run app.py` is required from the project root.

# Usage

<img width="1849" height="892" alt="image" src="https://github.com/user-attachments/assets/d254dcbb-3f67-4445-bdf9-afcb4ba651c7" />

You can select a pre-existing user on the left side of the screen. Each user exists in the original dataset and has their own set of preferences, reflected in the two tower network output. You also have the option of seeing output using only the two tower model and seeing it with the two tower model + LinUCB re-ranking system. Each item has a thumbs up and thumbs down button to simulate user interaction (clicks, purchases, reviews, etc.), which is used as input to the LinUCB contextual bandit.

# Architecture and decisions

The data flow starts in the `feature-pipeline.py` file. It takes in the raw CSV/JSON files from the dataset (including customer interactions and product descriptions), filters them, and transforms them into Parquet tables and optionally Delta Lake database. Parquet is used because its column-based storage reads significantly faster than CSVs and because it is the format that Delta Lake expects. Delta Lake and Spark are used because they have built-in systems for data integrity/incremental writes and parallel querying, respectively. These systems scale well because additional data can be appended to the database instead of rewriting the whole thing, and because the computational load of large queries can be divided among several processors.

Once the feature pipeline is built, the products are transformed into vector embeddings using a sentence tokenizer and saved with FAISS for fast vector-space computation at query time. This task is completed in `embed-products.py` and is run once at build time, like the feature pipeline. It is worth noting that, although these files are run once in this project, a production system would have them run on a schedule so that new user interactions and products can be reflected in the recommendations.
