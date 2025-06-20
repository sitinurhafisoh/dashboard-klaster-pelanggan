import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set tampilan awal
st.set_page_config(layout="wide", page_title="📊 Dashboard Data Mining", page_icon="📈")

# Judul Dashboard
st.title("📊 Dashboard Data Mining: Clustering & Regression 🚀")

# Deskripsi singkat
st.markdown("""
Selamat datang di dashboard interaktif untuk analisis data layanan menggunakan **metode Clustering (KMeans)** dan **Regresi Linear**.  
Dashboard ini bertujuan untuk membantu memahami pola layanan berdasarkan **frekuensi pemakaian dan total pendapatan**.
""")

# Load data
df = pd.read_excel("fact_servicerevenue_neww.xlsx")
df_selected = df[["Nama Layanan", "Total Service", "Total Pendapatan"]]
df_grouped = df_selected.groupby("Nama Layanan").agg({
    "Total Service": "sum",
    "Total Pendapatan": "sum"
}).reset_index()

# Clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_grouped[["Total Service", "Total Pendapatan"]])
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df_grouped["Cluster"] = kmeans.fit_predict(X_scaled)

# Regression
X = df_grouped[["Total Service"]]
y = df_grouped["Total Pendapatan"]
reg = LinearRegression()
reg.fit(X, y)
df_grouped["Predicted"] = reg.predict(X)
mse = mean_squared_error(y, df_grouped["Predicted"])
r2 = r2_score(y, df_grouped["Predicted"])

# Sidebar filter
st.sidebar.image("https://img.icons8.com/color/96/analytics.png", width=80)
st.sidebar.title("⚙️ Filter")
selected = st.sidebar.multiselect("Pilih Layanan", df_grouped["Nama Layanan"].unique(), default=df_grouped["Nama Layanan"].unique())
filtered_df = df_grouped[df_grouped["Nama Layanan"].isin(selected)]

# Plot Clustering
st.subheader("🎯 Clustering Layanan Berdasarkan Service & Pendapatan")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=filtered_df, x="Total Service", y="Total Pendapatan", hue="Cluster", palette="Set2", s=100, ax=ax1)
plt.xlabel("Total Service")
plt.ylabel("Total Pendapatan")
plt.title("Visualisasi Clustering")
st.pyplot(fig1)

# Plot Regression
st.subheader("📈 Regresi Linear: Total Service vs Pendapatan")
fig2, ax2 = plt.subplots()
sns.regplot(data=filtered_df, x="Total Service", y="Total Pendapatan", line_kws={"color": "red"}, ax=ax2)
plt.xlabel("Total Service")
plt.ylabel("Total Pendapatan")
plt.title("Visualisasi Regresi Linear")
st.pyplot(fig2)

# Evaluasi Model
st.subheader("🧪 Evaluasi Model Regresi")
st.markdown(f"""
- 🔢 **Mean Squared Error (MSE)**: `{mse:,.2f}`
- 📊 **R-squared Score (R²)**: `{r2:.4f}`
""")

# Data Tabel Akhir
st.subheader("📋 Data Akhir (Hasil Clustering & Prediksi)")
st.dataframe(filtered_df)
