# Laporan Proyek Machine Learning - Randy Cheasario Sam

## Domain Proyek

Industri minuman keras (liquor) merupakan salah satu sektor dengan tingkat distribusi yang kompleks dan dinamis. Perusahaan yang bergerak dalam penjualan wine, bir, dan spirits menghadapi tantangan dalam memprediksi permintaan pasar, terutama pada penjualan ritel. Ketersediaan data historis mengenai penjualan di gudang dan ritel memberikan peluang untuk membangun model prediktif dalam rangka mengoptimalkan stok dan strategi distribusi. Dengan memanfaatkan teknik Machine Learning, prediksi penjualan ritel dapat membantu perusahaan mengurangi biaya penyimpanan dan memaksimalkan profit.

**Referensi:**

1. Castaldelli-Maia, J. M., Segura, L. E., & Martins, S. S. (2021). *The concerning increasing trend of alcohol beverage sales in the U.S. during the COVID-19 pandemic*. Alcohol, 96, 37–42. https://doi.org/10.1016/j.alcohol.2021.06.004.
2. Paul, U. (2023). *COVID-19 Liquor Sales Forecasting Model*. In Lecture notes in computer science (pp. 499–505). https://doi.org/10.1007/978-3-031-47994-6_44

## Business Understanding

### Problem Statements

- Bagaimana memprediksi jumlah penjualan retail suatu produk berdasarkan data historis dan karakteristik produk?
- Apa saja fitur yang paling berpengaruh terhadap besarnya penjualan retail?

### Goals

- Membangun model prediksi yang mampu memperkirakan `RETAIL SALES` dengan akurasi tinggi.
- Mengidentifikasi variabel-variabel penting yang mempengaruhi penjualan retail untuk pengambilan keputusan bisnis.

### Solution Statements

- Menggunakan beberapa algoritma regresi seperti Random Forest Regressor, Linear Regression, dan Gradient Boosting Regressor.
- Menggunakan beberapa metrik evaluasi seperti Mean Absolute Error, Mean Squared Error, R² score dan Root Mean Squared Error (RMSE) untuk mengukur kinerja model.

## Data Understanding

Dataset berisi 307.645 baris data transaksi penjualan dari berbagai distributor minuman beralkohol.

**Variabel yang tersedia dalam dataset:**
- `YEAR`, `MONTH`: Tahun dan bulan transaksi
- `SUPPLIER`: Nama distributor atau manufaktur yang menyediakan alkohol
- `ITEM CODE`: Kode unik produk
- `ITEM DESCRIPTION`: Deskripsi produk
- `ITEM TYPE`: Jenis produk (misal: WINE, BEER, SPIRITS)
- `RETAIL SALES`: Jumlah penjualan unit di toko ritel ke pelanggan
- `RETAIL TRANSFERS`: Jumlah unit yang dipindahkan antar toko
- `WAREHOUSE SALES`: Jumlah unit yang dijual dari gudang ke toko ritel

**Exploratory Data Analysis (EDA)**

- Beberapa nilai missing ditemukan di kolom `SUPPLIER`, `ITEM TYPE`, dan `RETAIL SALES`, yang akan ditangani di tahap *data preparation*.
- Tidak ada data duplikat dalam dataset.
- Korelasi antara fitur `RETAIL SALES` dengan `RETAIL TRANSFERS` sangat tinggi dengan nilai **0.96** yang mengindikasikan jumlah transfer stok antar toko memiliki kaitan yang kuat dengan penjualan akhir ke konsumen.
- Variabel `RETAIL SALES`, `RETAIL TRANSFERS`, dan `WAREHOUSE SALES` memiliki outlier yang sangat banyak.
- Perusahaan pemasok minuman keras terbesar adalah `REPUBLIC NATIONAL DISTRIBUTING CO`.
- Tipe Item yang muncul paling banyak adalah `WINE`.
- Distribusi ketiga fitur numerik sangat right-skewed.
- Tahun `2019` adalah tahun dengan transaksi terbanyak.

## Data Preparation

1. Mengatasi missing values dengan mengisi kolom dengan modus untuk variabel kategorik dan menggunakan metode drop untuk menghapus missing values untuk variabel numerik.
2. Menggunakan metode scaling (MinMaxScaler) pada fitur numerik.
3. Menghindari outlier ekstrem melalui winsorization.

## Modeling

Pada proyek ini, saya menggunakan tiga algoritma regresi dengan variabel `RETAIL SALES` sebagai label. Berikut ini adalah algoritma yang digunakan:

**1. Linear Regression**
    - Mudah dipahami dan diinterpretasikan sehingga cocok sebagai baseline untuk mengetahui apakah hubungan antar fitur bersifat linier.
    - Sangat efisien untuk dataset besar dengani >300 ribu baris.
    
**2. Random Forest Regressor**
    - Mampu menangani data non-linear sehingga ideal untuk menangkap pola kompleks antar fitur.
    - Tahan terhadap outlier: Model tetap stabil meskipun ada nilai ekstrim atau noise.
    
**3. Gradient Boosting Regressor**
    - Akurasi tinggi dengan membangun model secara bertahap dengan memperbaiki kesalahan dari model sebelumnya.
    - Mengurangi bias dan varians, kombinasi boosting membuat model lebih presisi dibanding model tunggal.

## Evaluation

Metrik yang digunakan dalam eksperimen ini **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **R2 Score**, **Root Mean Squared Error (RMSE)**, dan **Mean CV R2 Score** karena sesuai dengan konteks regresi dan memberikan penalti besar untuk error yang signifikan.

1. **Mean Absolute Error (MAE)**

![MAE](/assets/mae.jpeg)

Mengukur rata-rata kesalahan absolut prediksi serta memberikan gambaran seberapa jauh prediksi dari nilai sebenarnya.

2. **Mean Squared Error (MSE)**

![MSE](/assets/mse.jpeg)

MSE adalah nilai rata-rata dari kuadrat kesalahan antara nilai sebenarnya dan nilai prediksi.

3. **R2 Score**

![R2 Score](/assets/r2_score.jpeg)

R<sup>2</sup> Score mengukur proporsi variasi dalam target yang dapat dijelaskan oleh model.

5. **Root Mean Squared Error (RMSE)**

![RMSE](/assets/rmse.png)

RMSE adalah akar kuadrat dari MSE. Ini mengembalikan kesalahan ke dalam satuan yang sama dengan data sehingga lebih mudah diinterpretasikan.

6. **Mean CV R2 Score**

![Mean CV R2 Score](/assets/mean_cv_r2_score.png)

Mean CV R<sup>2</sup> Score adalah nilai rata-rata performa dari semua fold CV, metrik ini dapat mengestimasi performa yang lebih robust sehingga dapat dijadikan dasar pemilihan model yang objektif.


### **Hasil dan Perbandingan**

| Model                      | MAE Score | MSE Score | R2 Score | RMSE    | Mean CV R2 Score |
|---------------------------|-----------|-----------|----------|---------|------------------|
| Random Forest Regressor   | 0.034014  | 0.005924  | 0.912827 | 0.076967| 0.913872         |
| Linear Regression         | 0.035551  | 0.005875  | 0.913540 | 0.076652| 0.914652         |
| Gradient Boosting Regressor | 0.033437| 0.005694  | 0.916209 | 0.075459| 0.917402         |


Gradient Boosting Regressor memiliki R<sup>2</sup> Score sebesar **0.9162** yang berarti model dapat menjelaskan lebih dari **91%** variasi dalam data. Gradient Boosting Regressor juga memiliki MAE terendah sebesar **0.0334**, hal tersebut menunjukkan prediksi rata-rata hanya meleset sekitar **3.3%** dari nilai sebenarnya. Selain itu, skor MSE dan RMSE dari Gradient Boosting Regressor paling rendah di antara semua model dengan skor masing-masing **0.00569** dan **0.0755**, menunjukan paling sedikit menghasilkan error besar dan sebaran error yang sangat terkontrol. Terakhir, rata-rata CV R<sup>2</sup> Score memiliki performa sebesar **91.74%** yang menunjukan model tidak overfitting..
