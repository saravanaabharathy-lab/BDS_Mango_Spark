# BDS Assignment 2: Retail Customer LTV with Hybrid MongoDB + Spark

CC ZG522 Big Data Systems, BITS Pilani WILP, First Semester 2026-2027
Scenario 1: Retail and E-Commerce (Omnichannel Analytics and Dynamic Customer LTV), Track C: Hybrid MongoDB + Spark
Student: Saravana Bharathy Anbazhagan (BITS ID 2026ns03034), individual assignment

MongoDB stores the nested order documents and runs the aggregation pipelines. Spark reads them through the
Mongo-Spark Connector, builds RFM segments and a repeat-purchase classifier (Random Forest and GBT), and writes
LTV estimates and model metrics back to MongoDB. A Streamlit and Plotly dashboard reads MongoDB live.

Data: Olist Brazilian E-Commerce Public Dataset (Kaggle), 99,441 orders from 2016 to 2018. It is not included here.

## Architecture

```
Olist CSVs --load_olist.py--> MongoDB retail.orders (nested orders)
                                  |  mongo_layer.js (aggregations, $merge)
                                  v
                          retail.customer_summary
                                  |  Mongo-Spark Connector 11.1.0
                                  v
                    Spark 4.1.2: features, RFM, RF and GBT, calibration
                                  |  write back
                                  v
              retail.customer_ltv, retail.model_metrics --> Streamlit dashboard (app.py)
```

## Environment (BITS Prayogshala lab, 3 nodes)

- MongoDB 8.0.29, Apache Spark 4.1.2 (Scala 2.13), Mongo-Spark Connector 11.1.0
- Python 3.12 (PySpark 4.1.2 does not import under the lab's default Python 3.9), numpy
- pymongo 4.19.0, pandas 3.1.1, Streamlit 1.67.0, Plotly 7.1.0, OpenJDK 17

## Run order

Adjust paths to your setup. The lab session used these steps:

1. Load the data: `python3.12 load_olist.py` (builds one nested document per order in `retail.orders` and creates the indexes)
2. Aggregation pipelines: `mongosh mongo_layer.js` (six pipelines, builds `retail.customer_summary`)
3. Spark job: `./run_spark.sh ltv_spark.py` (local[*], 4 GB driver, connector 11.1.0; writes `retail.customer_ltv` and `retail.model_metrics`)
4. Dashboard: `streamlit run app.py --server.port 8501`

## Files

| File | Purpose |
|---|---|
| `load_olist.py` | Reads the seven Olist CSVs, builds nested order documents, loads `retail.orders`, creates indexes |
| `mongo_layer.js` | Six aggregation pipelines ($group, $setWindowFields, $unwind, $bucket, $merge, $lookup, explain) |
| `ltv_spark.py`, `run_spark.sh` | Spark job and spark-submit wrapper |
| `app.py` | Streamlit and Plotly dashboard |
| `smoke.py` | Connector read and write test |
| `fix_calibration.js` | Probability calibration applied after the first Spark run, before it was built into `ltv_spark.py` |
| `bench_mongo.js`, `bench_spark.py` | Benchmark of four tasks in MongoDB and in Spark (five runs, median) |
| `run_cluster.sh`, `ltv_spark_cluster.py`, `bench_spark_cluster.py` | Two-worker Spark standalone cluster run (master spark://10.3.19.170:7077, 8 cores, 6 GB per executor) |
| `ltv_spark_local.py` | Control: the cluster script rerun in local[*], writing `*_local` collections |

## Key results

- 94,990 customers, 2,888 repeat buyers (3%), repeat base rate 4.03%
- GBT AUC 0.5844 (PR-AUC 0.0548), Random Forest AUC 0.5809 (PR-AUC 0.0529): a weak signal
- Class-weighted probabilities (mean about 0.44) are corrected back to the 4.03% base rate with the prior-correction formula
- Single-node rerun reproduces the metrics exactly; the two-node run changes the tree-model metrics and gives no speed-up at 99,441 documents

The full write-up, with screenshots and the benchmark charts, is in the submitted report.
