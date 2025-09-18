import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

# ======================
# Load Data
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("merged_retail_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.title("📊 Integrated Retail Analytics Dashboard")

# ======================
# Sidebar Filters
# ======================
st.sidebar.header("🔎 Filters")
store_id = st.sidebar.selectbox("Select Store", sorted(df['Store'].unique()))
dept_id = st.sidebar.selectbox("Select Department", sorted(df['Dept'].unique()))

# ======================
# 1. Exploratory Analysis
# ======================
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Weekly Sales Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Weekly_Sales'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Average Weekly Sales by Store Type")
    fig, ax = plt.subplots()
    sns.barplot(x="Type", y="Weekly_Sales", data=df, ax=ax)
    st.pyplot(fig)

# ======================
# 2. Anomaly Detection
# ======================
st.header("Anomaly Detection (Z-score > 3)")
df['Sales_Zscore'] = (df['Weekly_Sales'] - df['Weekly_Sales'].mean())/df['Weekly_Sales'].std()
anomalies = df[abs(df['Sales_Zscore']) > 3]

st.write(f"Detected anomalies: {anomalies.shape[0]}")
st.dataframe(anomalies.head())

# ======================
# 3. Time-Series Analysis
# ======================
st.header("📈 Seasonal Decomposition")
ts = df[(df['Store']==store_id) & (df['Dept']==dept_id)].set_index("Date")['Weekly_Sales']
ts = ts.resample("W").sum().fillna(method="ffill")

if len(ts) > 52:
    result = seasonal_decompose(ts, model="additive", period=52)
    fig = result.plot()
    st.pyplot(fig)
else:
    st.warning("Not enough data for seasonal decomposition (need > 52 weeks).")

# ======================
# 4. Customer Segmentation
# ======================
st.header("🛒 Store Segmentation (Clustering)")
store_group = df.groupby("Store").agg({
    "Weekly_Sales":"mean",
    "Size":"first",
    "Type":"first",
    "CPI":"mean",
    "Unemployment":"mean"
}).reset_index()

scaler = StandardScaler()
scaled = scaler.fit_transform(store_group[["Weekly_Sales","Size","CPI","Unemployment"]])

kmeans = KMeans(n_clusters=3, random_state=42)
store_group['Cluster'] = kmeans.fit_predict(scaled)

st.write(f"Silhouette Score: {silhouette_score(scaled, store_group['Cluster']):.2f}")

fig, ax = plt.subplots()
sns.scatterplot(x="Size", y="Weekly_Sales", hue="Cluster", data=store_group, palette="deep", ax=ax)
st.pyplot(fig)

# ======================
# 5. Demand Forecasting
# ======================
st.header("📊 Demand Forecasting (Holt-Winters)")
ts = df[(df['Store']==store_id) & (df['Dept']==dept_id)].set_index("Date")['Weekly_Sales']
ts = ts.resample("W").sum().fillna(method="ffill")

if len(ts) > 52:
    model = ExponentialSmoothing(ts, seasonal="add", seasonal_periods=52)
    fit = model.fit()
    forecast = fit.forecast(20)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(ts, label="Actual")
    ax.plot(forecast, label="Forecast", linestyle="--")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Not enough data for forecasting (need > 52 weeks).")

st.success("✅ Dashboard ready using merged_retail_dataset.csv")
