# Deteksi Warna Kulit Menggunakan K-Nearest Neighbor (KNN)

Ini adalah aplikasi web untuk melakukan segmentasi warna kulit secara real-time dari unggahan gambar atau aliran kamera langsung. Aplikasi ini menggunakan algoritma K-Nearest Neighbor (KNN) untuk mengklasifikasikan setiap piksel sebagai "kulit" atau "bukan kulit".

Proyek ini merupakan bagian dari "Minggu 2" dalam seri proyek "GLCM & KNN System", yang berfokus pada implementasi praktis algoritma *computer vision*.

## 📸 Tampilan Aplikasi

*(Tampilan utama aplikasi yang menunjukkan panel konfigurasi, panel hasil gambar, dan fitur deteksi kamera.)*

## ✨ Fitur Utama

* **Deteksi dari Unggahan File:** Pengguna dapat mengunggah gambar (JPG, PNG, dll.) untuk diproses.
* **Deteksi Live Camera:** Mampu menangkap gambar langsung dari webcam pengguna dan memprosesnya secara instan.
* **Konfigurasi Model yang Fleksibel:**
    * **Nilai K:** Memungkinkan pengguna memilih nilai K untuk algoritma KNN ($K = 1, 3, 5, 7$).
    * **Ruang Warna:** Mendukung pemrosesan dalam tiga ruang warna populer: **RGB**, **HSV**, dan **YCrCb**.
* **Visualisasi Hasil Ganda:** Menampilkan gambar asli dan hasil segmentasi (gambar *masking*) secara berdampingan untuk perbandingan yang mudah.
* **Evaluasi Model (ROC):** Secara dinamis menghasilkan kurva **ROC (Receiver Operating Characteristic)** dan menghitung nilai **AUC (Area Under Curve)** untuk mengevaluasi performa model pada gambar yang diproses.

## 🔬 Latar Belakang & Landasan Teori

Pemilihan algoritma **K-Nearest Neighbor (KNN)** untuk proyek ini didasarkan pada keefektifannya yang telah terbukti dalam penelitian akademis untuk klasifikasi warna kulit.

[cite_start]Sebuah studi oleh Syamsul Mujahidin (2015) [1] membandingkan beberapa algoritma klasifikasi, termasuk Bayesian, Multi Perceptrons, dan k-NN, pada dataset besar (245.057 sampel)[cite: 5, 58].

**Temuan Utama dari Penelitian:**
* [cite_start]**Kinerja Terbaik:** Algoritma **k-NN (IB1) menunjukkan performa tertinggi** dengan **akurasi 99.9559%** dan **presisi 99.937%**[cite: 92].
* [cite_start]**Efektivitas:** Penelitian ini menyimpulkan bahwa k-NN "sangat efektif digunakan" untuk aplikasi *computer vision* yang melibatkan data warna kulit[cite: 131].
* [cite_start]**Relevansi Ruang Warna:** Penelitian tersebut juga mengidentifikasi RGB [cite: 1][cite_start], HSV, dan YCbCr [cite: 23] sebagai ruang warna yang relevan untuk deteksi kulit, yang ketiganya diimplementasikan sebagai opsi dalam aplikasi ini.

[cite_start]Oleh karena itu, aplikasi ini mengimplementasikan metode (KNN) dan metrik evaluasi (ROC) [cite: 71] yang divalidasi oleh penelitian tersebut.

> [1] S. Mujahidin, "Klasifikasi Warna Kulit bedasarkan Ruang Warna RGB," *Seminar Nasional Aplikasi Teknologi Informasi (SNATi) 2015*, Yogyakarta, 6 Juni 2015.

## 💻 Tumpukan Teknologi

* **Frontend:** HTML, CSS, JavaScript (ES6+), Bootstrap 5
* **Visualisasi Data:** Chart.js
* **Backend (Dipersumsikan):** Python (Flask/Django) melalui endpoint `fetch` ke `/skin_process` dan `/skin_camera_process`.
* **Library (Dipersumsikan):** OpenCV (untuk pemrosesan gambar), Scikit-learn (untuk implementasi KNN dan metrik ROC/AUC).

## 🚀 Cara Penggunaan

### 1. Deteksi dari File
1.  Pilih nilai **K** (misal: 3).
2.  Pilih **Ruang Warna** (misal: RGB).
3.  Klik "Pilih File" (tombol `Choose File`) dan unggah gambar Anda.
4.  Klik tombol **"Proses Deteksi"**.
5.  Hasil "Gambar Asli" dan "Hasil Segmentasi" akan muncul di panel kanan.

### 2. Deteksi dari Kamera
1.  Pastikan konfigurasi (K dan Ruang Warna) sudah sesuai.
2.  Arahkan wajah atau tangan Anda ke video *feed* kamera.
3.  Klik tombol **"Ambil & Deteksi"**.
4.  Hasil akan muncul di bawah *feed* kamera.

### 3. Evaluasi Kinerja
1.  Setelah memproses gambar (baik dari file atau kamera), data ROC akan tersimpan.
2.  Klik tombol **"Tampilkan Evaluasi"** di panel kiri.
3.  Sebuah kurva ROC dan nilai AUC (Area Under Curve) akan ditampilkan di bawah tombol tersebut.

## (Dipersumsikan) Instalasi & Penyiapan Lokal

1.  **Clone repositori:**
    ```bash
    git clone https://[URL_REPOSITORI_ANDA].git
    cd [NAMA_FOLDER_REPOSITORI]
    ```
2.  **Siapkan Backend (Python):**
    * Buat dan aktifkan *virtual environment*:
        ```bash
        python -m venv venv
        source venv/bin/activate  # (atau venv\Scripts\activate di Windows)
        ```
    * Install *dependencies* (misal, dalam `requirements.txt`):
        ```bash
        pip install flask opencv-python scikit-learn numpy
        ```
    * Jalankan server backend:
        ```bash
        python app.py
        ```
3.  **Jalankan Frontend:**
    * Buka file `index.html` (atau nama file HTML utama Anda) di browser.
    * Aplikasi akan secara otomatis terhubung ke server backend yang berjalan di `localhost`.

---