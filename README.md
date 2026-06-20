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

`feature-pipeline.py` - uses raw CSV/JSON files from dataset and transforms them into Parquet tables and a Delta Lake database. Parquet is used because its column-based storage reads significantly faster than CSVs and because it is the format that Delta Lake expects. Delta Lake and Spark are used because they have built-in systems for data integrity/incremental writes and parallel querying, respectively. These systems scale well because additional data can be appended to the database instead of rewriting the whole thing, and because the computational load of large queries can be divided among several processors. For this project, the metadata is overwritten to the feature store every time the pipeline is run, however in a production system this would be replaced with an upsert to add more products on the fly.

User features include session data and category affinity. Item features include rating statistics and purchase frequency.

`embed-products.py` - transforms product data from Amazon dataset into vector embeddings with a sentence tokenizer and saved with FAISS for fast vector-space computation at query time. This task is completed in `embed-products.py` and is run once at build time, like the feature pipeline. It is worth noting that, although these files are run once in this project, a production system would have them run on a schedule so that new user interactions and products can be reflected in the recommendations.

# Next steps

As with any AI-native project, the code needs to be reviewed and tweaked to be its most efficient. Although I prioritized learning over production speed, I still need to take more steps to understand the lower-level components of this code so I can maintain architectural ownership. The overall structure fits together, but legacy code and folder structure needs to be improved. The data loading must be optimized to reduce the amount of data sent through the API and minimize the amount of compute required to serve the models. I am presently completing these tasks.
