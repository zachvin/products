"""
Feature Pipeline
================
Reads raw interactions + item metadata, computes user and item features,
and writes two Delta Lake tables:

    ./feature_store/user_features
    ./feature_store/item_features

Run:
    python feature_pipeline.py

Requirements:
    pip install pyspark delta-spark pandas pyarrow
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window
from pyspark.sql.types import StringType, StructType, StructField, ArrayType
import os

# ---------------------------------------------------------------------------
# 1. Spark session with Delta Lake support
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("feature-pipeline")
    .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    # Keep memory reasonable for a laptop
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Open train/valid/test CSV files
train = spark.read.option("header", True).option("inferSchema", True).csv("data/raw/Cell_Phones_and_Accessories.train.csv")
valid = spark.read.option("header", True).option("inferSchema", True).csv("data/raw/Cell_Phones_and_Accessories.valid.csv")
test  = spark.read.option("header", True).option("inferSchema", True).csv("data/raw/Cell_Phones_and_Accessories.test.csv")

# Rename parent_asin to item_id
train = train.withColumnRenamed("parent_asin", "item_id")
valid = valid.withColumnRenamed("parent_asin", "item_id")
test  = test.withColumnRenamed("parent_asin", "item_id")

# Save each split as Parquet
train.write.mode("overwrite").parquet("./data/processed/train.parquet")
valid.write.mode("overwrite").parquet("./data/processed/valid.parquet")
test.write.mode("overwrite").parquet("./data/processed/test.parquet")

# Combine all three for the feature pipeline
all_interactions = train.union(valid).union(test)
all_interactions.write.mode("overwrite").parquet("./data/processed/interactions.parquet")

# Save metadata json file
# Explicit schema avoids schema-inference errors caused by duplicate-cased keys
# in the `details` field (e.g. "Assembly Required" vs "assembly required").
_meta_schema = StructType([
    StructField("parent_asin",   StringType(),              True),
    StructField("main_category", StringType(),              True),
    StructField("title",         StringType(),              True),
    StructField("price",         StringType(),              True),
    StructField("features",      ArrayType(StringType()),   True),
    StructField("description",   ArrayType(StringType()),   True),
    StructField("categories",    ArrayType(StringType()),   True),
])
item_ids = all_interactions.select("item_id").distinct()

metadata = (
    spark.read.schema(_meta_schema).json("./data/raw/meta_Cell_Phones_and_Accessories.jsonl")
    .join(item_ids, on=F.col("parent_asin") == F.col("item_id"), how="inner")
    .select(
        "parent_asin",
        "main_category",
        "title",
        "price",
        # Join array fields into a single space-separated string
        F.concat_ws(" ", F.col("features")).alias("features"),
        F.concat_ws(" ", F.col("description")).alias("description"),
        F.concat_ws(" ", F.col("categories")).alias("categories"),
    )
)

metadata.write.mode("overwrite").parquet("./data/processed/metadata.parquet")

# ---------------------------------------------------------------------------
# 2. Load data
#    Assumes you have already saved filtered interactions and metadata as
#    Parquet files.  Adjust paths if yours differ.
# ---------------------------------------------------------------------------

INTERACTIONS_PATH = "./data/processed/interactions.parquet"
METADATA_PATH     = "./data/processed/metadata.parquet"
FEATURE_STORE     = "./feature_store"

print("Loading interactions...")
interactions = spark.read.parquet(INTERACTIONS_PATH)
# Expected columns: user_id (string), item_id (string),
#                   rating (float), timestamp (long),
#                   verified_purchase (boolean)

print("Loading metadata...")
metadata = spark.read.parquet(METADATA_PATH)
# Expected columns: parent_asin (string), main_category (string),
#                   title (string), price (string)

# ---------------------------------------------------------------------------
# 3. Basic cleaning
# ---------------------------------------------------------------------------
# Shouldn't be necessary, but will include basic cleaning anyway
interactions = (
    interactions
    .filter(F.col("user_id").isNotNull())
    .filter(F.col("item_id").isNotNull())
    .filter(F.col("rating").between(1.0, 5.0))
    .filter(F.col("timestamp").isNotNull())
)

# Normalise price to a numeric column on the metadata side
metadata = metadata.withColumn(
    "price_numeric",
    F.regexp_replace(F.col("price"), "[^0-9.]", "").try_cast("float"),
)

# ---------------------------------------------------------------------------
# 4. Enrich interactions with category from metadata
# ---------------------------------------------------------------------------

interactions_enriched = interactions.join(
    metadata.select("parent_asin", "main_category", "price_numeric"),
    interactions["item_id"] == metadata["parent_asin"],
    how="left",
)

# ---------------------------------------------------------------------------
# 5. Session features
#    A session = consecutive interactions by the same user with < 30-min gaps.
# ---------------------------------------------------------------------------

SESSION_GAP_SECONDS = 30 * 60  # 30 minutes

user_time_window = Window.partitionBy("user_id").orderBy("timestamp")

interactions_with_sessions = (
    interactions_enriched
    # Time gap to previous interaction for this user
    .withColumn("prev_ts", F.lag("timestamp").over(user_time_window))
    .withColumn(
        "gap_seconds",
        F.col("timestamp") - F.col("prev_ts"),
    )
    # Flag the start of each new session
    .withColumn(
        "is_new_session",
        F.when(
            F.col("gap_seconds").isNull() | (F.col("gap_seconds") > SESSION_GAP_SECONDS),
            1,
        ).otherwise(0),
    )
    # Assign a session ID = cumulative sum of session-start flags
    .withColumn(
        "session_id",
        F.sum("is_new_session").over(user_time_window),
    )
)

# Aggregate to session level
session_agg = interactions_with_sessions.groupBy("user_id", "session_id").agg(
    F.count("*").alias("session_length"),
    F.avg("rating").alias("session_avg_rating"),
    F.min("timestamp").alias("session_start_ts"),
    F.max("timestamp").alias("session_end_ts"),
)

session_agg = session_agg.withColumn(
    "session_duration_mins",
    (F.col("session_end_ts") - F.col("session_start_ts")) / 60.0,
)

# Roll up to user level
user_session_features = session_agg.groupBy("user_id").agg(
    F.count("session_id").alias("total_sessions"),
    F.avg("session_length").alias("avg_session_length"),
    F.max("session_length").alias("max_session_length"),
    F.avg("session_duration_mins").alias("avg_session_duration_mins"),
    F.max("session_duration_mins").alias("max_session_duration_mins"),
)

# ---------------------------------------------------------------------------
# 6. Rating pattern features  (user level)
# ---------------------------------------------------------------------------

rating_features = interactions_enriched.groupBy("user_id").agg(
    F.count("*").alias("total_purchases"),
    F.avg("rating").alias("avg_rating"),
    F.stddev("rating").alias("rating_stddev"),
    F.sum(F.when(F.col("rating") >= 4.0, 1).otherwise(0)).alias("high_rating_count"),
    F.sum(F.when(F.col("rating") <= 2.0, 1).otherwise(0)).alias("low_rating_count"),
    F.max("timestamp").alias("last_purchase_ts"),
    F.min("timestamp").alias("first_purchase_ts"),
)

# Days since last purchase (relative to the most recent timestamp in dataset)
max_ts = interactions_enriched.agg(F.max("timestamp")).collect()[0][0]

rating_features = rating_features.withColumn(
    "days_since_last_purchase",
    (max_ts - F.col("last_purchase_ts")) / 86400.0,
).withColumn(
    "days_active",
    (F.col("last_purchase_ts") - F.col("first_purchase_ts")) / 86400.0,
).withColumn(
    "purchase_frequency",
    # avg days between purchases; guard against divide-by-zero
    F.when(
        F.col("total_purchases") > 1,
        F.col("days_active") / (F.col("total_purchases") - 1),
    ).otherwise(F.lit(None).try_cast("float")),
)

# ---------------------------------------------------------------------------
# 7. Category affinity  (user level)
# ---------------------------------------------------------------------------

# Count purchases per (user, category)
category_counts = interactions_enriched.groupBy("user_id", "main_category").agg(
    F.count("*").alias("category_purchase_count")
)

# Rank categories per user; take the top one
category_rank_window = Window.partitionBy("user_id").orderBy(
    F.desc("category_purchase_count")
)

favorite_category = (
    category_counts
    .withColumn("cat_rank", F.rank().over(category_rank_window))
    .filter(F.col("cat_rank") == 1)
    .select("user_id", F.col("main_category").alias("favorite_category"))
)

# Diversity: how many distinct categories does the user shop across?
category_diversity = interactions_enriched.groupBy("user_id").agg(
    F.countDistinct("main_category").alias("distinct_categories")
)

# Average price of items purchased (proxy for price sensitivity)
price_sensitivity = interactions_enriched.groupBy("user_id").agg(
    F.avg("price_numeric").alias("avg_item_price"),
    F.max("price_numeric").alias("max_item_price"),
)

# ---------------------------------------------------------------------------
# 8. Join all user features together
# ---------------------------------------------------------------------------

print("Assembling user features...")

user_features = (
    rating_features
    .join(favorite_category,   on="user_id", how="left")
    .join(category_diversity,  on="user_id", how="left")
    .join(price_sensitivity,   on="user_id", how="left")
    .join(user_session_features, on="user_id", how="left")
)

# Replace nulls in numeric cols with sensible defaults
numeric_cols = [
    "rating_stddev", "avg_session_length", "max_session_length",
    "avg_session_duration_mins", "max_session_duration_mins",
    "avg_item_price", "max_item_price", "purchase_frequency",
    "days_active",
]
for col in numeric_cols:
    user_features = user_features.fillna({col: 0.0})

user_features = user_features.fillna({"favorite_category": "unknown"})

# ---------------------------------------------------------------------------
# 9. Item features
# ---------------------------------------------------------------------------

print("Computing item features...")

item_features = interactions_enriched.groupBy("item_id").agg(
    F.count("*").alias("total_purchases"),
    F.countDistinct("user_id").alias("unique_buyers"),
    F.avg("rating").alias("avg_rating"),
    F.stddev("rating").alias("rating_stddev"),
    F.count("rating").alias("rating_count"),
    F.sum(F.when(F.col("rating") >= 4.0, 1).otherwise(0)).alias("high_rating_count"),
    F.max("timestamp").alias("last_purchased_ts"),
)

item_features = item_features.withColumn(
    "days_since_last_purchased",
    (max_ts - F.col("last_purchased_ts")) / 86400.0,
)

# Join in metadata fields
item_features = item_features.join(
    metadata.select(
        F.col("parent_asin").alias("item_id"),
        "main_category",
        "price_numeric",
    ),
    on="item_id",
    how="left",
)

item_features = item_features.fillna({"rating_stddev": 0.0, "price_numeric": 0.0})

# ---------------------------------------------------------------------------
# 10. Write to Delta Lake
# ---------------------------------------------------------------------------

user_out = os.path.join(FEATURE_STORE, "user_features")
item_out = os.path.join(FEATURE_STORE, "item_features")

print(f"Writing user features to {user_out} ...")
(
    user_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(user_out)
)

print(f"Writing item features to {item_out} ...")
(
    item_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(item_out)
)

# ---------------------------------------------------------------------------
# 11. Sanity checks
# ---------------------------------------------------------------------------

print("\n=== User features sample ===")
spark.read.format("delta").load(user_out).show(5, truncate=False)

print("\n=== Item features sample ===")
spark.read.format("delta").load(item_out).show(5, truncate=False)

print("\n=== Row counts ===")
n_users = spark.read.format("delta").load(user_out).count()
n_items = spark.read.format("delta").load(item_out).count()
print(f"  User feature rows : {n_users:,}")
print(f"  Item feature rows : {n_items:,}")

print("\nPipeline complete.")
spark.stop()