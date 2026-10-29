# 🐾 PawScan — Deteksi Kucing dan Anjing Berbasis AI

**PawScan** adalah aplikasi berbasis web yang dirancang untuk mengenali kucing dan anjing menggunakan teknologi *Artificial Intelligence* (AI) dan *Computer Vision*. Aplikasi ini menyediakan dua metode analisis gambar, yaitu **Convolutional Neural Network (CNN)** untuk klasifikasi gambar dan **YOLO** untuk deteksi objek.

PawScan dikembangkan menggunakan Python dan Streamlit dengan antarmuka biru-putih yang responsif, sehingga mudah digunakan melalui laptop maupun perangkat seluler.

## ✨ Fitur Utama

- 🐱 **Klasifikasi Gambar dengan CNN**  
  Menganalisis gambar untuk memprediksi kelas hewan berdasarkan model klasifikasi yang digunakan.

- 🐶 **Deteksi Objek dengan YOLO**  
  Mendeteksi lokasi objek kucing dan anjing dalam gambar menggunakan *bounding box* serta menampilkan label dan nilai confidence hasil deteksi.

- 🖼️ **Unggah Gambar**  
  Memungkinkan pengguna mengunggah gambar untuk dianalisis.

- 📊 **Ringkasan Hasil Analisis**  
  Menampilkan hasil prediksi atau deteksi dalam antarmuka aplikasi.

- 🎨 **Antarmuka Responsif**  
  Menggunakan desain biru-putih dengan ilustrasi maskot, kartu langkah, dan ikon bertema hewan.

## 🛠️ Teknologi yang Digunakan

- **Python** — bahasa pemrograman utama.
- **Streamlit** — framework untuk membangun aplikasi web.
- **CNN (Convolutional Neural Network)** — metode klasifikasi gambar.
- **YOLO (You Only Look Once)** — metode deteksi objek.
- **Pillow, NumPy, dan OpenCV** — pengolahan gambar, sesuai penggunaan dalam kode aplikasi.

## 📁 Struktur Proyek

```text
deteksi-kucing-anjing/
├── app.py
├── requirements.txt
├── assets/
│   ├── maskot_pawscan.png
│   ├── kartu_01_registrasi.png
│   ├── kartu_02_pilih_metode.png
│   ├── kartu_03_unggah_hasil.png
│   └── paw_print_blue.png
└── model/
    └── best.pt
```

*Catatan: Struktur di atas merupakan contoh. Sesuaikan nama dan lokasi file dengan struktur repository yang sebenarnya. Model CNN juga dapat disimpan di lokasi berbeda.*

## ⚙️ Instalasi

### 1. Clone repository

```bash
git clone https://github.com/muhammadputhe/deteksi-kucing-anjing.git
cd deteksi-kucing-anjing
```

### 2. Buat virtual environment (opsional)

```bash
python -m venv venv
```

Aktifkan environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Instal dependensi

Jika repository menyediakan `requirements.txt`, jalankan:

```bash
pip install -r requirements.txt
```

Pastikan seluruh pustaka yang dibutuhkan oleh aplikasi dan model sudah terpasang.

### 4. Siapkan model

Letakkan file model CNN dan YOLO pada lokasi yang sesuai dengan konfigurasi di `app.py`. Pastikan nama file dan path model cocok dengan yang digunakan aplikasi.

### 5. Jalankan aplikasi

```bash
streamlit run app.py
```

Setelah server berjalan, buka alamat lokal yang ditampilkan pada terminal untuk menggunakan PawScan.

## 🚀 Cara Menggunakan PawScan

1. Buka aplikasi PawScan.
2. Isi informasi registrasi jika diminta.
3. Pilih metode analisis **CNN** atau **YOLO**.
4. Unggah gambar kucing atau anjing.
5. Jalankan analisis.
6. Lihat hasil klasifikasi atau deteksi yang ditampilkan aplikasi.

## 🧠 Perbedaan CNN dan YOLO

| Aspek | CNN | YOLO |
|---|---|---|
| Tujuan | Klasifikasi gambar | Deteksi objek |
| Hasil utama | Kelas prediksi | Label dan lokasi objek |
| Bounding box | Tidak selalu digunakan | Digunakan untuk menandai objek |
| Keluaran | Prediksi sesuai kelas model | Kotak deteksi dan confidence |

Hasil analisis bergantung pada kualitas gambar, data pelatihan, konfigurasi, dan model yang digunakan.

## 📌 Catatan

- Pastikan file model tersedia sebelum menjalankan aplikasi.
- Gunakan gambar yang jelas agar objek lebih mudah dikenali.
- Nilai confidence merupakan tingkat keyakinan prediksi model, bukan jaminan bahwa hasil selalu benar.
- Kinerja model perlu dievaluasi menggunakan data pengujian yang sesuai.

## 🔮 Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

- Meningkatkan kualitas dan keragaman dataset.
- Mengevaluasi model menggunakan metrik yang sesuai.
- Mengoptimalkan tampilan untuk perangkat seluler.
- Menambahkan riwayat analisis dan fitur ekspor hasil.
- Meningkatkan penanganan gambar yang tidak dapat dikenali model.

## 👨‍💻 Pengembang

**PawScan — Smart Pet Recognition**

Repository: [muhammadputhe/deteksi-kucing-anjing](https://github.com/muhammadputhe/deteksi-kucing-anjing)

---

*Dikembangkan sebagai proyek penerapan kecerdasan buatan dan computer vision untuk klasifikasi serta deteksi kucing dan anjing.* 🐾
