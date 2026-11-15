import time, statistics
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("bench")
         .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1:27017")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
CID = "7c396fd4830fd04220f754e42b4e5bff"

def load():
    return (spark.read.format("mongodb")
            .option("database", "retail").option("collection", "orders").load())

def valid(df):
    return df.filter(~F.col("status").isin("canceled", "unavailable"))

def bench(label, fn, runs=5):
    ts, rows = [], None
    for _ in range(runs):
        t = time.perf_counter(); rows = fn(); ts.append(time.perf_counter() - t)
    med = statistics.median(ts)
    print(f"BENCH spark | {label:30s} | median {med:.3f}s | runs {', '.join('%.3f' % x for x in ts)} | rows {rows}")

t = time.perf_counter(); n = load().count()
print(f"BENCH spark | warm-up (first read, {n} docs) | {time.perf_counter() - t:.3f}s")

tasks = {
 "A revenue by category": lambda d: valid(d).select(F.explode("items").alias("i"))
        .groupBy("i.category").agg(F.sum("i.price").alias("rev")).orderBy(F.desc("rev")).count(),
 "B monthly revenue": lambda d: valid(d)
        .groupBy(F.date_format("purchase_ts", "yyyy-MM").alias("m"))
        .agg(F.sum("order_value").alias("rev")).orderBy("m").count(),
 "C one customer's orders": lambda d: d.filter(F.col("customer.unique_id") == CID).count(),
 "D customer roll-up": lambda d: valid(d).groupBy("customer.unique_id")
        .agg(F.count("*").alias("n"), F.sum("order_value").alias("spend")).count(),
}

print("BENCH --- Spark reading from MongoDB on every run (connector read included)")
for label, fn in tasks.items():
    bench(label, lambda fn=fn: fn(load()))

print("BENCH --- Spark with orders cached in memory (read once, then reuse)")
cached = load().cache(); cached.count()
for label, fn in tasks.items():
    bench(label + " [cached]", lambda fn=fn: fn(cached))
spark.stop()
