from pyspark.sql import SparkSession
spark = (SparkSession.builder.appName("smoke")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1:27017")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1:27017")
    .getOrCreate())
df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
df.write.format("mongodb").mode("overwrite").option("database", "smoke").option("collection", "t").save()
spark.read.format("mongodb").option("database", "smoke").option("collection", "t").load().show()
spark.stop()
