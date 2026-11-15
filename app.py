
import pandas as pd, plotly.express as px, streamlit as st

from pymongo import MongoClient



st.set_page_config(page_title="Retail LTV - Hybrid Mongo + Spark", layout="wide")

db = MongoClient("mongodb://127.0.0.1:27017")["retail"]



@st.cache_data

def load():

    ltv = pd.DataFrame(db.customer_ltv.find({}, {"_id": 0}))

    met = pd.DataFrame(db.model_metrics.find({}, {"_id": 0}))

    trend = pd.DataFrame(db.orders.aggregate([

        {"$match": {"status": {"$nin": ["canceled", "unavailable"]},

                    "purchase_ts": {"$gte": pd.Timestamp("2017-01-01").to_pydatetime(),

                                    "$lt": pd.Timestamp("2018-09-01").to_pydatetime()}}},

        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m", "date": "$purchase_ts"}},

                    "orders": {"$sum": 1}, "revenue": {"$sum": "$order_value"}}},

        {"$sort": {"_id": 1}}])).rename(columns={"_id": "month"})

    trend["rolling_3m"] = trend["revenue"].rolling(3, min_periods=1).mean()

    cats = pd.DataFrame(db.orders.aggregate([

        {"$match": {"status": {"$nin": ["canceled", "unavailable"]}}}, {"$unwind": "$items"},

        {"$group": {"_id": "$items.category", "revenue": {"$sum": "$items.price"},

                    "units": {"$sum": 1}}},

        {"$sort": {"revenue": -1}}, {"$limit": 10}])).rename(columns={"_id": "category"})

    return ltv, met, trend, cats



ltv, met, trend, cats = load()

st.title("Retail Omnichannel Analytics - Customer LTV")

st.caption("Olist e-commerce | MongoDB (operational store) + PySpark (RFM, GBT/RF) via Mongo-Spark Connector")



segs = st.sidebar.multiselect("Segments", sorted(ltv["segment"].unique()),

                              default=sorted(ltv["segment"].unique()))

states = st.sidebar.multiselect("States", sorted(ltv["state"].dropna().unique()))

f = ltv[ltv["segment"].isin(segs)]

if states:

    f = f[f["state"].isin(states)]



c1, c2, c3, c4 = st.columns(4)

c1.metric("Customers", f"{len(f):,}")

c2.metric("Total spend", f"R$ {f['total_spend'].sum():,.0f}")

c3.metric("Avg LTV estimate", f"R$ {f['ltv_estimate'].mean():,.1f}")

c4.metric("Repeat buyers", f"{(f['orders'] >= 2).mean() * 100:.1f}%")



a, b = st.columns(2)

a.plotly_chart(px.line(trend, x="month", y=["revenue", "rolling_3m"],

               title="Monthly revenue and 3-month rolling average (Mongo $setWindowFields)"),

               use_container_width=True)

b.plotly_chart(px.bar(cats, x="revenue", y="category", orientation="h",

               title="Top 10 categories by revenue (Mongo $unwind)").update_yaxes(autorange="reversed"),

               use_container_width=True)



seg = (f.groupby("segment").agg(customers=("segment", "size"), avg_spend=("total_spend", "mean"),

       repeat_prob=("repeat_prob", "mean"), avg_ltv=("ltv_estimate", "mean")).reset_index())

c, d = st.columns(2)

c.plotly_chart(px.bar(seg, x="segment", y="avg_ltv", color="segment",

               title="Average LTV by RFM segment (Spark)"), use_container_width=True)

d.plotly_chart(px.pie(seg, names="segment", values="customers", title="Customer mix by segment"),

               use_container_width=True)



e, g = st.columns(2)

e.plotly_chart(px.histogram(f[f["ltv_estimate"] < 1000], x="ltv_estimate", nbins=50,

               title="LTV estimate distribution (under R$1000)"), use_container_width=True)

g.subheader("Model comparison (repeat-purchase classifier)")

g.dataframe(met, use_container_width=True)

g.caption("Class-weighted training, prior-corrected probabilities. AUC is modest by design: "

          "first-order attributes only, no leakage.")



st.subheader("Top 15 customers by LTV estimate")

st.dataframe(f.sort_values("ltv_estimate", ascending=False).head(15), use_container_width=True)

