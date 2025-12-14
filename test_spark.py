try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("Test").getOrCreate()
    print("SUCCESS! PySpark is working.")
    print("Spark Version:", spark.version)
    spark.stop()
except Exception as e:
    print("FAILED! Error:", e)
