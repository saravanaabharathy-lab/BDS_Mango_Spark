from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

URI = "mongodb://10.3.19.170:27017"
spark = (SparkSession.builder.appName("retail_ltv_cluster")
         .config("spark.mongodb.read.connection.uri", URI)
         .config("spark.mongodb.write.connection.uri", URI)
         .config("spark.sql.ansi.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

def rd(coll):
    return (spark.read.format("mongodb")
            .option("database", "retail").option("collection", coll).load())

# ---------- 1. Read nested orders from MongoDB, flatten in Spark ----------
orders = rd("orders")
o = (orders
     .filter(~F.col("status").isin("canceled", "unavailable"))
     .filter((F.col("purchase_ts") >= "2017-01-01") & (F.col("purchase_ts") < "2018-09-01"))
     .select(F.col("_id").alias("order_id"),
             F.col("customer.unique_id").alias("cust"),
             F.col("customer.state").alias("state"),
             "purchase_ts", "delivered_ts", "estimated_delivery_ts",
             "order_value", "freight_total", "review_score",
             F.size("items").alias("n_items"),
             F.col("items")[0]["category"].alias("category"),
             F.col("payments")[0]["payment_type"].alias("pay_type"),
             F.array_max(F.transform("payments",
                         lambda p: p["payment_installments"])).alias("installments")))
print("orders in reliable window:", o.count())

# ---------- 2. First-order features + repeat label ----------
w = Window.partitionBy("cust").orderBy("purchase_ts")
o = o.withColumn("rn", F.row_number().over(w))
stats = o.groupBy("cust").agg(F.count("*").alias("n_orders"),
                              F.min("purchase_ts").alias("first_ts"))
first = (o.filter("rn = 1").join(stats, "cust")
         .withColumn("repeat", (F.col("n_orders") >= 2).cast("int"))
         .withColumn("delay_days", F.datediff("delivered_ts", "estimated_delivery_ts"))
         .fillna({"review_score": 0, "installments": 1, "delay_days": 0,
                  "n_items": 1, "pay_type": "unknown", "state": "unknown"}))

top = [r[0] for r in first.groupBy("category").count()
       .orderBy(F.desc("count")).limit(20).collect() if r[0] is not None]
first = first.withColumn("cat_grp", F.when(F.col("category").isin(top), F.col("category"))
                                     .otherwise("other")).cache()

cohort = first.filter(F.col("first_ts") < "2018-03-01")
base = cohort.agg(F.avg("repeat")).first()[0]
print("cohort customers:", cohort.count(), "| repeat rate: %.4f" % base)
cohort = cohort.withColumn("weight", F.when(F.col("repeat") == 1, (1 - base) / base)
                                      .otherwise(1.0))

# ---------- 3. MLlib pipeline: Random Forest vs GBT ----------
cat_cols = ["state", "pay_type", "cat_grp"]
num_cols = ["order_value", "freight_total", "n_items", "installments",
            "review_score", "delay_days"]
prep = [StringIndexer(inputCol=c, outputCol=c + "_i", handleInvalid="keep") for c in cat_cols]
prep.append(OneHotEncoder(inputCols=[c + "_i" for c in cat_cols],
                          outputCols=[c + "_v" for c in cat_cols], handleInvalid="keep"))
prep.append(VectorAssembler(inputCols=num_cols + [c + "_v" for c in cat_cols],
                            outputCol="features"))

train, test = cohort.randomSplit([0.8, 0.2], seed=42)
auc = BinaryClassificationEvaluator(labelCol="repeat", metricName="areaUnderROC")
pr = BinaryClassificationEvaluator(labelCol="repeat", metricName="areaUnderPR")

models = {
    "RandomForest": RandomForestClassifier(labelCol="repeat", weightCol="weight",
                                           numTrees=100, maxDepth=8, seed=42),
    "GBT": GBTClassifier(labelCol="repeat", weightCol="weight",
                         maxIter=40, maxDepth=5, seed=42),
}
results, fitted = [], {}
for name, est in models.items():
    m = Pipeline(stages=prep + [est]).fit(train)
    pred = m.transform(test)
    a, p = auc.evaluate(pred), pr.evaluate(pred)
    fitted[name] = m
    results.append((name, float(a), float(p), float(train.count()), float(test.count())))
    print("%-13s AUC=%.4f  PR-AUC=%.4f" % (name, a, p))

best = max(results, key=lambda r: r[1])[0]
print("best model:", best)
imp = fitted[best].stages[-1].featureImportances.toArray()
print("numeric feature importances:",
      sorted(zip(num_cols, [round(float(x), 3) for x in imp[:len(num_cols)]]),
             key=lambda t: -t[1]))

# ---------- 4. Score everyone, build RFM + LTV ----------
scored = (fitted[best].transform(first)
          .select("cust", vector_to_array("probability")[1].alias("repeat_prob_raw"))
          .withColumn("repeat_prob",
              (F.col("repeat_prob_raw") * base) /
              (F.col("repeat_prob_raw") * base + (1 - F.col("repeat_prob_raw")) * (1 - base))))
aov = o.agg(F.avg("order_value")).first()[0]

cs = rd("customer_summary")
rfm = (cs.select(F.col("_id").alias("cust"), "state", "orders", "total_spend",
                 "last_purchase", "avg_review")
       .withColumn("recency_days",
                   F.greatest(F.datediff(F.lit("2018-09-01").cast("date"),
                                         F.col("last_purchase")), F.lit(0)))
       .withColumn("r_score", F.ntile(4).over(Window.orderBy(F.desc("recency_days"))))
       .withColumn("m_score", F.ntile(4).over(Window.orderBy("total_spend")))
       .withColumn("f_score", F.when(F.col("orders") >= 3, 3).otherwise(F.col("orders"))))

seg = (F.when((F.col("f_score") >= 2) & (F.col("r_score") >= 3), "Loyal Active")
        .when(F.col("f_score") >= 2, "Loyal Lapsed")
        .when((F.col("m_score") == 4) & (F.col("r_score") >= 3), "Big Spender Recent")
        .when(F.col("m_score") == 4, "Big Spender Lapsed")
        .when(F.col("r_score") >= 3, "New / Recent")
        .otherwise("Lapsed One-time"))

final = (rfm.join(scored, "cust", "left")
         .fillna({"repeat_prob": float(base)})
         .withColumn("segment", seg)
         .withColumn("expected_future_value", F.col("repeat_prob") * F.lit(float(aov)))
         .withColumn("ltv_estimate", F.col("total_spend") + F.col("expected_future_value"))
         .withColumnRenamed("cust", "_id")
         .drop("last_purchase"))

(final.groupBy("segment")
 .agg(F.count("*").alias("customers"), F.round(F.avg("total_spend"), 1).alias("avg_spend"),
      F.round(F.avg("repeat_prob"), 4).alias("avg_repeat_prob"),
      F.round(F.avg("ltv_estimate"), 1).alias("avg_ltv"))
 .orderBy(F.desc("customers")).show(truncate=False))

# ---------- 5. Persist predictions back to MongoDB ----------
(final.write.format("mongodb").mode("overwrite")
 .option("database", "retail").option("collection", "customer_ltv_cluster").save())
spark.createDataFrame(results, ["model", "auc", "pr_auc", "train_rows", "test_rows"]) \
     .write.format("mongodb").mode("overwrite") \
     .option("database", "retail").option("collection", "model_metrics_cluster").save()
print("written: retail.customer_ltv =", final.count(), "rows; retail.model_metrics")
spark.stop()
