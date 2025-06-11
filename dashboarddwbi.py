import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Judul Aplikasi
st.title("📊 Dashboard Analisis Layanan dan Pelanggan Salon")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file: fact_servicerevenue_new.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧼 Data Asli")
    st.dataframe(df.head())

    # ==============================
    # 📌 BAGIAN 1: Analisis Layanan Paling Laku
    # ==============================
    st.header("🔥 Analisis Layanan Paling Laku (Berdasarkan Total Service)")

    st.markdown("""
    Tujuan analisis ini adalah mengetahui layanan **yang paling sering digunakan pelanggan**, sebagai dasar untuk:
    - Strategi promosi
    - Pengembangan layanan serupa
    - Pengalokasian sumber daya salon
    """)

    top_services = df.groupby('Nama Layanan')['Total Service'].sum().reset_index()
    top_services = top_services.sort_values(by='Total Service', ascending=False)

    st.subheader("🏆 Tabel Layanan Terlaris")
    st.dataframe(top_services.head(10).reset_index(drop=True))

    st.subheader("📊 Visualisasi: Total Service per Layanan")
    fig2 = px.bar(
        top_services.sort_values(by='Total Service'),
        x='Total Service',
        y='Nama Layanan',
        orientation='h',
        title='Top Layanan Berdasarkan Jumlah Digunakan',
        color='Total Service',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig2)

    # ==============================
    # 📌 BAGIAN 2: Clustering Pelanggan
    # ==============================
    st.header("👥 Clustering Pelanggan Berdasarkan Preferensi")

    customer_summary = df.groupby('ID Pelanggan').agg({
        'Total Pendapatan': 'sum',
        'Total Service': 'sum',
        'Harga (IDR)': 'mean'
    }).reset_index()
    customer_summary.columns = ['ID Pelanggan', 'Total Pendapatan', 'Total Service', 'Rata-rata Harga Layanan']

    scaler = StandardScaler()
    features = customer_summary[['Total Pendapatan', 'Total Service', 'Rata-rata Harga Layanan']]
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_summary['Cluster'] = kmeans.fit_predict(scaled_features)
    silhouette = silhouette_score(scaled_features, customer_summary['Cluster'])

    st.markdown(f"**✅ Silhouette Score:** `{silhouette:.2f}` (Semakin mendekati 1, semakin baik pemisahan klasternya)")

    fig = px.scatter_3d(
        customer_summary,
        x='Total Pendapatan',
        y='Total Service',
        z='Rata-rata Harga Layanan',
        color='Cluster',
        hover_data=['ID Pelanggan'],
        title='Visualisasi Clustering Pelanggan'
    )
    st.plotly_chart(fig)

    st.subheader("📊 Rata-rata Setiap Fitur per Cluster")
    st.dataframe(
        customer_summary.groupby('Cluster')[['Total Pendapatan', 'Total Service', 'Rata-rata Harga Layanan']]
        .mean()
        .round(2)
    )

else:
    st.info("Silakan upload file CSV untuk memulai analisis.")
