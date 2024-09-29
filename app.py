import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
import io

# Set Streamlit page configuration
st.set_page_config(page_title="DASHBOARD DATASET WISATA", layout="wide")

# Page Header
st.header('STREAMLIT DATASET WISATA JAWA BARAT')
st.write('50421643_Ilham Rizkyansyah_Kelas C')

# Load dataset
with st.expander('DATA UNDERSTANDING'):
    st.write('Memuat dataset')
    file_path = 'disparbud-od_15387_jml_ptns_obyek_daya_tarik_wisata_odtw__jenis_kabup_v2_data.csv'
    data = pd.read_csv(file_path)
    st.write(data)

    st.write('Menampilkan informasi dasar tentang dataset')
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write('Menampilkan 5 baris pertama dari Dataset')
    st.write(data.head())

    st.write('Menampilkan statistik deskriptif dari dataset')
    st.write(data.describe(include='all'))

    # Check for missing values
    st.write('Cek nilai yang hilang (NaN) pada dataset:')
    missing_values = data.isnull().sum()
    st.write(missing_values)

# Data Preparation Section
with st.expander('DATA PREPARATION'):
    st.write('Menangani data dan mempersiapkannya untuk analisis.')

    # Handle missing values by filling with 0
    st.write('Mengisi nilai yang hilang dengan 0.')
    data = data.fillna(0)
    st.write('Dataset setelah menangani nilai yang hilang:')
    st.write(data.head())

    # Label Encoding for categorical variables if needed
    st.write('Encoding kolom kategorikal jika ada.')
    le = LabelEncoder()
    data['nama_provinsi'] = le.fit_transform(data['nama_provinsi'])
    st.write('Dataset setelah encoding:')
    st.write(data.head())

    # Outlier Trimming using Interquartile Range (IQR) for 'jumlah_odtw'
    st.write('Trimming outliers dari jumlah obyek daya tarik wisata menggunakan metode IQR.')
    Q1 = data['jumlah_odtw'].quantile(0.25)
    Q3 = data['jumlah_odtw'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['jumlah_odtw'] >= lower_bound) & (data['jumlah_odtw'] <= upper_bound)]
    st.write('Dataset setelah trimming outliers:')
    st.write(data.describe())

# Data Visualization Section
with st.expander('DATA VISUALIZATION'):
    st.write('Visualisasi data jumlah obyek daya tarik wisata berdasarkan jenis.')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['jenis_odtw'], ax=ax)
    st.pyplot(fig)

    st.write('Visualisasi distribusi jumlah ODTW.')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=data['jumlah_odtw'], ax=ax)
    st.pyplot(fig)

# Machine Learning Models Section
with st.expander('MODELING'):
    st.write('Penerapan Model Linear Regression untuk prediksi jumlah obyek daya tarik wisata.')

    # Feature selection and target
    X = data[['kode_provinsi', 'kode_kabupaten_kota']]  # Adjust feature columns as needed
    y = data['jumlah_odtw']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation with additional metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write('Evaluation Metrics:')
    st.write('Mean Absolute Error (MAE):', mae)
    st.write('Mean Squared Error (MSE):', mse)
    st.write('Root Mean Squared Error (RMSE):', rmse)
    st.write('R2 Score:', r2)

    # Visualization of Predictions
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    st.pyplot(fig)

# Clustering Section
with st.expander('K-MEANS CLUSTERING'):
    st.write('Clustering menggunakan K-Means berdasarkan jumlah ODTW.')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['jumlah_odtw']])

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_data)

    st.write('Hasil clustering:')
    st.write(data.head())

    # Visualization of Clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='kode_kabupaten_kota', y='jumlah_odtw', hue='cluster', data=data, palette='viridis', ax=ax)
    st.pyplot(fig)

# Success message
st.success('run success')
