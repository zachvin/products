# Products

This repository is an AI-native project that builds a product recommendation system based on the [Amazon Reviews](https://amazon-reviews-2023.github.io/) dataset. I built it to learn more about scalable, production-ready ML systems with real applications.

# Architecture and decisions

The data flow starts in the `feature-pipeline.py` file. It takes in the raw CSV/JSON files from the dataset (including customer interactions and product descriptions), filters them, and transforms them into Parquet tables and optionally Delta Lake database. Parquet is used because its column-based storage reads significantly faster than CSVs and because it is the format that Delta Lake expects. Delta Lake and Spark are used because they have built-in systems for data integrity/incremental writes and parallel querying, respectively. These systems scale well because additional data can be appended to the database instead of rewriting the whole thing, and because the computational load of large queries can be divided among several processors.

Once the feature pipeline is built, the products are transformed into vector embeddings using a sentence tokenizer and saved with FAISS for fast vector-space computation at query time. This task is completed in `embed-products.py` and is run once at build time, like the feature pipeline. It is worth noting that, although these files are run once in this project, a production system would have them run on a schedule so that new user interactions and products can be reflected in the recommendations.
