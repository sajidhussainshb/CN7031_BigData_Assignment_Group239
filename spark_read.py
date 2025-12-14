from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# --------------------------------------------------------------------------------
# 1. Spark Session
# --------------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("IMDB-UDF-Task") \
    .getOrCreate()

# --------------------------------------------------------------------------------
# 2. Load the cleaned IMDB dataset
# Replace the path with your actual file path
# --------------------------------------------------------------------------------
df = spark.read.csv("clean_imdb_reviews.csv", header=True, inferSchema=True)

# df.show(5)

# --------------------------------------------------------------------------------
# 3. Define UDF to categorize sentiment
# --------------------------------------------------------------------------------
def classify_sentiment(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

sentiment_udf = udf(classify_sentiment, StringType())

# --------------------------------------------------------------------------------
# 4. Apply UDF
# --------------------------------------------------------------------------------
df_with_class = df.withColumn("sentiment_category", sentiment_udf(df["sentiment_score"]))

df_with_class.show(10)

# --------------------------------------------------------------------------------
# 5. Save the results (optional but good for assignment)
# --------------------------------------------------------------------------------
df_with_class.write.mode("overwrite").csv("imdb_with_sentiment_category")
