import os, pandas as pd
from pymongo import MongoClient

P = os.path.expanduser("~/workspace/olist/")
rd = lambda f: pd.read_csv(P + f)

orders = rd("olist_orders_dataset.csv")
cust   = rd("olist_customers_dataset.csv")
items  = rd("olist_order_items_dataset.csv")
pay    = rd("olist_order_payments_dataset.csv")
rev    = rd("olist_order_reviews_dataset.csv")
prod   = rd("olist_products_dataset.csv")
trans  = rd("product_category_name_translation.csv")

# product category in English
prod = prod.merge(trans, on="product_category_name", how="left")
prod["category"] = (prod["product_category_name_english"]
                    .fillna(prod["product_category_name"]).fillna("unknown"))

items = items.merge(prod[["product_id", "category"]], on="product_id", how="left")
items_g = {k: g[["order_item_id", "product_id", "seller_id", "category",
                 "price", "freight_value"]].to_dict("records")
           for k, g in items.groupby("order_id")}
pay_g = {k: g[["payment_sequential", "payment_type", "payment_installments",
               "payment_value"]].to_dict("records")
         for k, g in pay.groupby("order_id")}
rev_d = (rev.sort_values("review_creation_date")
            .drop_duplicates("order_id", keep="last")
            .set_index("order_id")["review_score"].to_dict())

orders = orders.merge(cust, on="customer_id", how="left")
dates = ["order_purchase_timestamp", "order_approved_at",
         "order_delivered_customer_date", "order_estimated_delivery_date"]
for c in dates:
    orders[c] = pd.to_datetime(orders[c])

def ts(v):
    return None if pd.isna(v) else v.to_pydatetime()

docs = []
for r in orders.itertuples(index=False):
    its = items_g.get(r.order_id, [])
    docs.append({
        "_id": r.order_id,
        "status": r.order_status,
        "purchase_ts": ts(r.order_purchase_timestamp),
        "approved_ts": ts(r.order_approved_at),
        "delivered_ts": ts(r.order_delivered_customer_date),
        "estimated_delivery_ts": ts(r.order_estimated_delivery_date),
        "customer": {"unique_id": r.customer_unique_id, "city": r.customer_city,
                     "state": r.customer_state, "zip": int(r.customer_zip_code_prefix)},
        "items": its,
        "payments": pay_g.get(r.order_id, []),
        "review_score": rev_d.get(r.order_id),
        "order_value": float(sum(i["price"] for i in its)),
        "freight_total": float(sum(i["freight_value"] for i in its)),
    })

db = MongoClient("mongodb://127.0.0.1:27017")["retail"]
db.orders.drop()
for i in range(0, len(docs), 5000):
    db.orders.insert_many(docs[i:i + 5000])

db.orders.create_index("customer.unique_id")
db.orders.create_index("purchase_ts")
db.orders.create_index("status")
print("orders loaded:", db.orders.count_documents({}))
