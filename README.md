# Prediksi Depresi Mahasiswa (Student Depression Prediction)

Proyek ini bertujuan untuk memprediksi tingkat/risiko depresi pada mahasiswa menggunakan pendekatan *Machine Learning*, berdasarkan dataset dari Kaggle. Proyek ini dilengkapi dengan dashboard interaktif berbasis Streamlit untuk melakukan inference (prediksi) secara langsung.

---

## Daftar Isi
- [Problem Identification](#-problem-identification) 
- [Dataset](#-dataset)
- [Data Processing](#-data-processing)
- [Model & Hasil Evaluasi](#-model--hasil-evaluasi)
- [Struktur Folder](#-struktur-folder)
- [Cara Menjalankan (Local Setup)](#-cara-menjalankan-local-setup)
- [Demo Aplikasi](#-demo-aplikasi)

---

## Problem Identification

**Latar Belakang:**
Mahasiswa berada pada masa transisi dari remaja akhir menuju dewasa awal, sebuah fase yang membuat mereka rentan terhadap gangguan kesehatan mental akibat tekanan akademik, sosial, dan ekonomi. Hal ini didukung oleh studi *Gambaran Kondisi Kesehatan Mental Mahasiswa* (Arfandi et al., 2025) yang dipublikasikan di *Journal of Mental Health Concerns* (Vol. 4, No. 2, 2025), yang melakukan survei terhadap 275 mahasiswa dari 13 perguruan tinggi di Kota Samarinda menggunakan kuesioner DASS-21. Hasil penelitian tersebut menunjukkan bahwa sekitar 30% mahasiswa mengalami gejala depresi dari kategori ringan hingga berat, dengan 9,8% di antaranya berada pada kategori depresi berat. Studi yang sama juga menemukan bahwa mahasiswa perempuan memiliki prevalensi depresi lebih tinggi (34,4%) dibanding laki-laki (26,7%), dan mahasiswa pada semester awal cenderung lebih rentan mengalami depresi (41%) dibanding semester pertengahan maupun akhir, yang dikaitkan dengan proses adaptasi terhadap lingkungan kampus dan sistem pembelajaran baru.

Temuan ini memperkuat urgensi proyek ini: gangguan kesehatan mental, khususnya depresi, bukan isu kecil di kalangan mahasiswa, namun deteksi dini secara manual melalui instrumen seperti DASS-21 membutuhkan waktu, tenaga ahli, dan proses administratif yang tidak selalu bisa menjangkau seluruh mahasiswa secara cepat dan masif.

**Mengapa AI relevan untuk masalah ini?**
Faktor-faktor risiko depresi pada mahasiswa — seperti tekanan akademik, kepuasan studi, durasi tidur, pola makan, stres finansial, dan jam belajar — saling berinteraksi secara kompleks dan tidak selalu bersifat linear, sehingga sulit dianalisis hanya dengan observasi manual. Pendekatan *Machine Learning* memungkinkan sistem mempelajari pola dari data historis mahasiswa untuk mengklasifikasikan risiko depresi secara objektif, konsisten, dan cepat. Dengan demikian, sistem berbasis AI ini dapat berfungsi sebagai alat bantu **skrining awal** yang melengkapi instrumen seperti DASS-21, sehingga institusi pendidikan dapat lebih sigap mengidentifikasi mahasiswa berisiko sebelum kondisi mereka memburuk, sejalan dengan rekomendasi penelitian di atas yang menekankan pentingnya layanan dukungan psikologis serta program promotif dan preventif di lingkungan kampus.

> **Referensi:**
> Arfandi, M. A., Rahman, R. A., Gah, R. L., Asma, N., Mahendra, A. Z., & Rosandini, A. N. (2025). Gambaran kondisi kesehatan mental mahasiswa di Kota Samarinda, Kalimantan Timur. *Journal of Mental Health Concerns*, 4(2), 102–111. https://doi.org/10.56922/mhc.v4i2.1194

---
## Dataset

- **Sumber:** Kaggle (file `depression.csv`) — (https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- **Fitur yang digunakan:**
  - `Gender`
  - `Age`
  - `Academic Pressure`
  - `Study Satisfaction`
  - `Sleep Duration`
  - `Dietary Habits`
  - `Financial Stress`
  - `Work/Study Hours`
- **Target/Label:** `Depression` (0 = Tidak Depresi, 1 = Depresi)

---

## Data Processing

Tahapan yang dilakukan terhadap data sebelum pemodelan:

1. **Data Cleaning**
   - Mengecek nilai kosong (`isnull().sum()`) dan mengisi nilai kosong pada kolom numerik menggunakan **median** (`fillna(df.median())`).
   - Menyeleksi 9 kolom relevan yang digunakan dalam pemodelan (lihat daftar fitur di atas).
   - Mengecek dan menghapus data duplikat (`drop_duplicates()`).
2. **Encoding**
   - Mengubah seluruh kolom bertipe kategorikal (object) menjadi numerik menggunakan **Label Encoding** (`sklearn.preprocessing.LabelEncoder`).
3. **Exploratory Data Analysis (EDA)**
   - Distribusi target `Depression` (countplot).
   - Perbandingan `Gender` terhadap `Depression`.
   - Hubungan `Academic Pressure` dan `Financial Stress` terhadap `Depression` (boxplot).
   - Distribusi `Age` mahasiswa (histogram + KDE).
   - Seluruh visualisasi EDA dapat dilihat pada notebook `Proyek_Prediksi_Depresi.ipynb`.

---

## Model & Hasil Evaluasi

Dilakukan eksperimen komparatif antara 2 algoritma klasifikasi pada dataset yang sama (split data 80% train, 20% test, `random_state=0`):

| Model | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
|---|---|---|---|---|
| Decision Tree | 0.69 | 0.68 | 0.68 | 0.68 |
| **Random Forest** | **0.77** | **0.77** | **0.76** | **0.76** |
<img width="580" height="203" alt="image" src="https://github.com/user-attachments/assets/e2460c96-1cf7-4d1c-aeff-eb3005584de2" />
<img width="581" height="197" alt="image" src="https://github.com/user-attachments/assets/a9649cc4-1013-4fac-9c4f-c4688783c51c" />

**Analisis Singkat:**
Model **Random Forest** memberikan performa yang lebih baik dibanding Decision Tree, dengan kenaikan akurasi dari 69% menjadi 77%. Hal ini wajar karena Random Forest menggabungkan banyak pohon keputusan (*ensemble*) sehingga lebih tahan terhadap *overfitting* dan mampu menangkap pola hubungan antar fitur yang lebih kompleks dibanding satu Decision Tree saja. Berdasarkan analisis *feature importance* dari Random Forest, faktor-faktor seperti tekanan akademik dan jam belajar/kerja menjadi salah satu fitur dengan pengaruh terbesar terhadap prediksi risiko depresi (lihat detail urutan importance di notebook). Model akhir (`rf_model`) disimpan dalam file `model_depression.pkl` menggunakan `joblib` untuk selanjutnya digunakan pada aplikasi Streamlit.

<img width="408" height="86" alt="image" src="https://github.com/user-attachments/assets/b7c5d371-cc5c-49b8-bd9c-05bb8afae3c0" />

---

## Struktur Folder

```
dashboard/
│
├── Proyek_Prediksi_Depresi.ipynb   # Notebook EDA, preprocessing, training & evaluasi model
├── depression.csv                  # Dataset mentah dari Kaggle
├── model_depression.pkl            # Model Random Forest hasil training (joblib)
├── app.py                          # File utama dashboard Streamlit
├── requirements.txt                # Daftar library + versi yang dibutuhkan
└── README.md
```

Lokasi project di komputer lokal:
```
C:\Users\ASUS\OneDrive - ITPLN\Documents\dashboard
```

---
## Cara Menjalankan (Local Setup)

Proyek ini dijalankan menggunakan **Anaconda Prompt**. Ikuti langkah - langkah berikut:

1. Prasyarat

Pastikan kamu sudah menginstal Anaconda (sudah termasuk Python di dalamnya).

2. Clone Repository

Buka terminal atau command prompt anda, lalu jalankan:
> Jika project belum ada di komputer dan masih di GitHub, clone dulu:
> ``` 
> git clone https://github.com/kamaliakurniasari92-cyber/Kamalia-Kurniasari_202432005.git
> cd Kamalia-Kurniasari_202432005
> ```

3. Buat Environment Anaconda Khusus untuk Project Ini

```
conda create -n depresi-env python=3.11 -y
conda activate depresi-env
```
> Setelah aktif, di depan baris perintah akan berubah jadi `(depresi-env)`. Setiap kali ingin menjalankan dashboard ini secara lokal, jalankan dulu `conda activate depresi-env` sebelum lanjut ke langkah berikutnya.

4. Install Library yang Dibutuhkan

```
pip install -r requirements.txt
```

5. Jalankan Dashboard
``` 
streamlit run app.py
``` 
Setelah perintah di atas dijalankan, dashboard akan otomatis terbuka di browser pada alamat:
``` 
http://localhost:8501
```
Catatan: Jika ingin melihat proses training model dari awal (EDA, preprocessing, hingga evaluasi), buka file Proyek_Prediksi_Depresi.ipynb menggunakan Jupyter Notebook:
``` 
jupyter notebook Proyek_Prediksi_Depresi.ipynb
```

## Demo Aplikasi

Dashboard dibangun menggunakan **Streamlit** dan **Plotly**, terdiri dari 3 halaman utama yang dapat diakses melalui sidebar navigasi:

1. **🏠 Home**
   
   <img width="959" height="500" alt="image" src="https://github.com/user-attachments/assets/e528c3f3-5941-4913-8dcf-acea351cb54d" />
   Menampilkan ringkasan proyek: jumlah data, jumlah fitur, serta perbandingan akurasi Decision Tree (69%) vs Random Forest (77%) dalam bentuk bar chart interaktif, lengkap dengan detail *classification report* tiap model.

   a. Grafik Perbandingan Decision Tree & Random Forest
       <img width="959" height="502" alt="image" src="https://github.com/user-attachments/assets/cc51a92b-ab98-49cc-bef5-fd02b5ca8b7e" />
   b. Classification Report Tiap Model
       <img width="959" height="502" alt="image" src="https://github.com/user-attachments/assets/9c693fc0-287d-4fb6-9ab1-9030d1d98a7d" />
      
2. **📊 Analytics Dashboard**
   
   <img width="959" height="500" alt="image" src="https://github.com/user-attachments/assets/325da331-7238-4225-85fa-379dd130f226" />
   Berisi visualisasi eksploratif interaktif: distribusi label depresi (pie chart), distribusi usia, perbandingan gender vs depresi, distribusi tekanan akademik & stres finansial, *feature importance* Random Forest, serta *confusion matrix* untuk masing-masing model.

   a. Distribusi Depresi
       <img width="959" height="499" alt="image" src="https://github.com/user-attachments/assets/17e78c8b-6c18-41a2-a403-6b462a150864" />
       
   b. Distribusi Usia
       <img width="959" height="500" alt="image" src="https://github.com/user-attachments/assets/a0db5062-653f-4e09-8623-7677263f260c" />
       
   c. Perbandingan Gender dan Usia
       <img width="959" height="500" alt="image" src="https://github.com/user-attachments/assets/9e76b15c-57e2-4be2-808e-270bd4748d7a" />
       
   d. Distribusi Tekanan Akademik
       <img width="959" height="502" alt="image" src="https://github.com/user-attachments/assets/beb7ae12-1d5b-49f5-bbbf-56f850ad81b7" />
       
   e. Distribusi Stres Finansial
       <img width="959" height="500" alt="image" src="https://github.com/user-attachments/assets/208f4967-ec4e-4ea1-a3cd-c5660ee3e729" />
       
   f. Feature Importance (Random Forest)
       <img width="959" height="499" alt="image" src="https://github.com/user-attachments/assets/0bee4482-8171-42f5-81e0-d6ac297007fb" />
       
   g. Confusion Matrix Decision Tree
       <img width="959" height="499" alt="image" src="https://github.com/user-attachments/assets/c8b217d4-b1cf-4e9e-9f05-65f133b1bbf0" />
       
   h. Confusion Matrix Random Forest
       <img width="959" height="497" alt="image" src="https://github.com/user-attachments/assets/4ddef2a9-1b97-4df9-b322-78ab6ed66fb0" />

3. **🔍 Prediction**
   
   <img width="959" height="503" alt="image" src="https://github.com/user-attachments/assets/e74412a0-5733-4ab8-ae9a-c6adda1b6f77" />
   Form input data mahasiswa baru (gender, usia, tekanan akademik, kepuasan studi, durasi tidur, pola makan, stres finansial, jam belajar) untuk melakukan inference langsung menggunakan model `model_depression.pkl`. Hasil prediksi ditampilkan beserta persentase probabilitas risiko dan rekomendasi tindak lanjut.

   Tampilan ketika hasil prediksinya berisiko tinggi depresi:
   
   <img width="614" height="240" alt="image" src="https://github.com/user-attachments/assets/fdef4505-0411-4b9e-ae13-50dc78ae7bf6" />

   Tampilan ketika hasil prediksinya berisiko rendah depresi:
   
   <img width="608" height="238" alt="image" src="https://github.com/user-attachments/assets/3dbb654a-e639-43b9-8420-315a2f861079" />

**🌐 Live Demo:** https://kamalia-kurniasari202432005-project.streamlit.app/

> Catatan: Aplikasi dijalankan secara online melalui Streamlit Community Cloud, dan juga dapat dijalankan secara lokal mengikuti panduan pada bagian [Cara Menjalankan (Local Setup)](#-cara-menjalankan-local-setup) di atas.

---
- **Kamalia Kurniasari — 202432005**
