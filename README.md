# Kamalia-Kurniasari_202432005
# Prediksi Depresi Mahasiswa (Student Depression Prediction)

Proyek ini bertujuan untuk memprediksi tingkat/risiko depresi pada mahasiswa menggunakan pendekatan *Machine Learning*, berdasarkan dataset dari Kaggle. Proyek ini dilengkapi dengan dashboard interaktif berbasis Streamlit untuk melakukan inference (prediksi) secara langsung.

---

## 📌 Daftar Isi
- [Problem Identification](#-problem-identification)
- [Dataset](#-dataset)
- [Data Processing](#-data-processing)
- [Model & Hasil Evaluasi](#-model--hasil-evaluasi)
- [Struktur Folder](#-struktur-folder)
- [Cara Menjalankan (Local Setup)](#-cara-menjalankan-local-setup)
- [Demo Aplikasi](#-demo-aplikasi)
- [Kontributor](#-kontributor)

---

## 🧠 Problem Identification

**Latar Belakang:**
Mahasiswa berada pada masa transisi dari remaja akhir menuju dewasa awal, sebuah fase yang membuat mereka rentan terhadap gangguan kesehatan mental akibat tekanan akademik, sosial, dan ekonomi. Hal ini didukung oleh studi *Gambaran Kondisi Kesehatan Mental Mahasiswa* (Arfandi et al., 2025) yang dipublikasikan di *Journal of Mental Health Concerns* (Vol. 4, No. 2, 2025), yang melakukan survei terhadap 275 mahasiswa dari 13 perguruan tinggi di Kota Samarinda menggunakan kuesioner DASS-21. Hasil penelitian tersebut menunjukkan bahwa sekitar 30% mahasiswa mengalami gejala depresi dari kategori ringan hingga berat, dengan 9,8% di antaranya berada pada kategori depresi berat. Studi yang sama juga menemukan bahwa mahasiswa perempuan memiliki prevalensi depresi lebih tinggi (34,4%) dibanding laki-laki (26,7%), dan mahasiswa pada semester awal cenderung lebih rentan mengalami depresi (41%) dibanding semester pertengahan maupun akhir, yang dikaitkan dengan proses adaptasi terhadap lingkungan kampus dan sistem pembelajaran baru.

Temuan ini memperkuat urgensi proyek ini: gangguan kesehatan mental, khususnya depresi, bukan isu kecil di kalangan mahasiswa, namun deteksi dini secara manual melalui instrumen seperti DASS-21 membutuhkan waktu, tenaga ahli, dan proses administratif yang tidak selalu bisa menjangkau seluruh mahasiswa secara cepat dan masif.

**Mengapa AI relevan untuk masalah ini?**
Faktor-faktor risiko depresi pada mahasiswa — seperti tekanan akademik, kepuasan studi, durasi tidur, pola makan, stres finansial, dan jam belajar — saling berinteraksi secara kompleks dan tidak selalu bersifat linear, sehingga sulit dianalisis hanya dengan observasi manual. Pendekatan *Machine Learning* memungkinkan sistem mempelajari pola dari data historis mahasiswa untuk mengklasifikasikan risiko depresi secara objektif, konsisten, dan cepat. Dengan demikian, sistem berbasis AI ini dapat berfungsi sebagai alat bantu **skrining awal** yang melengkapi instrumen seperti DASS-21, sehingga institusi pendidikan dapat lebih sigap mengidentifikasi mahasiswa berisiko sebelum kondisi mereka memburuk, sejalan dengan rekomendasi penelitian di atas yang menekankan pentingnya layanan dukungan psikologis serta program promotif dan preventif di lingkungan kampus.

> **Referensi:**
> Arfandi, M. A., Rahman, R. A., Gah, R. L., Asma, N., Mahendra, A. Z., & Rosandini, A. N. (2025). Gambaran kondisi kesehatan mental mahasiswa di Kota Samarinda, Kalimantan Timur. *Journal of Mental Health Concerns*, 4(2), 102–111. https://doi.org/10.56922/mhc.v4i2.1194

---
## 📊 Dataset

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

## 🧹 Data Processing

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

## 🤖 Model & Hasil Evaluasi

Dilakukan eksperimen komparatif antara 2 algoritma klasifikasi pada dataset yang sama (split data 80% train, 20% test, `random_state=0`):

| Model | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
|---|---|---|---|---|
| Decision Tree | 0.69 | 0.68 | 0.68 | 0.68 |
| **Random Forest** | **0.77** | **0.77** | **0.76** | **0.76** |
<img width="580" height="203" alt="image" src="https://github.com/user-attachments/assets/e2460c96-1cf7-4d1c-aeff-eb3005584de2" />
<img width="581" height="197" alt="image" src="https://github.com/user-attachments/assets/a9649cc4-1013-4fac-9c4f-c4688783c51c" />

