import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Real Estate Price Estimator (Starter)", layout="wide")
st.title("🏠 AI Real Estate Price Estimator (Starter)")

st.write(\"\"\"
Upload a small CSV of properties to train a quick model, or generate a synthetic demo dataset.
Recommended columns: **bedrooms, bathrooms, sqft, year_built, zipcode, price**.
\"\"\")

def make_synthetic(n=500, seed=42):
    rng = np.random.default_rng(seed)
    sqft = rng.integers(500, 5000, n)
    beds = np.clip((sqft/700 + rng.normal(0,0.8,n)).round().astype(int), 1, 7)
    baths = np.clip((beds - 1 + rng.normal(0,0.4,n)).round().astype(int), 1, 5)
    year = rng.integers(1950, 2023, n)
    zipc = rng.choice([60601,60602,60603,60604,60605,60606,60607], n)
    base = 120 * sqft + 8000 * beds + 5000 * baths + 300 * (year-1950)
    noise = rng.normal(0, 50000, n)
    price = np.clip(base + noise, 80000, 3000000)
    return pd.DataFrame(dict(bedrooms=beds, bathrooms=baths, sqft=sqft, year_built=year, zipcode=zipc, price=price))

with st.sidebar:
    use_demo = st.checkbox("Use synthetic demo dataset", value=True)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RandomForest trees", 50, 500, 200, 50)
    max_depth = st.slider("Max depth", 3, 30, 12, 1)
    st.markdown("---")
    st.markdown("### Predict a New Property")
    p_sqft = st.number_input("sqft", value=1500, min_value=300, max_value=10000, step=50)
    p_beds = st.number_input("bedrooms", value=3, min_value=0, max_value=10, step=1)
    p_baths = st.number_input("bathrooms", value=2, min_value=0, max_value=10, step=1)
    p_year = st.number_input("year_built", value=2005, min_value=1800, max_value=2025, step=1)
    p_zip = st.selectbox("zipcode", options=[60601,60602,60603,60604,60605,60606,60607], index=0)

uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

if use_demo or uploaded is None:
    df = make_synthetic()
else:
    df = pd.read_csv(uploaded)

df.columns = [c.lower() for c in df.columns]
req_cols = {"bedrooms","bathrooms","sqft","year_built","zipcode","price"}
if not req_cols.issubset(set(df.columns)):
    st.error(f"Dataset must include columns: {sorted(req_cols)}")
    st.stop()

X = pd.get_dummies(df[["bedrooms","bathrooms","sqft","year_built","zipcode"]], columns=["zipcode"], drop_first=True)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

st.subheader("Model Performance")
st.metric("MAE (test)", f"${mae:,.0f}")
st.bar_chart(pd.DataFrame({
    "Actual": y_test.reset_index(drop=True).head(100),
    "Predicted": pd.Series(preds).head(100)
}))

st.subheader("Feature Importances")
fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.dataframe(fi.to_frame("importance"))

st.subheader("Predict a Property")
new_row = pd.DataFrame([{
    "bedrooms": p_beds,
    "bathrooms": p_baths,
    "sqft": p_sqft,
    "year_built": p_year,
    "zipcode": p_zip
}])
new_row = pd.get_dummies(new_row, columns=["zipcode"], drop_first=True)
new_row = new_row.reindex(columns=X.columns, fill_value=0)
pred_price = model.predict(new_row)[0]
st.success(f"Estimated price: **${pred_price:,.0f}**")

st.caption("Educational starter app. Upload your own CSV to improve estimates. Not financial advice.")