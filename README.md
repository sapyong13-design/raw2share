# RAW2Share

RAW2Share adalah aplikasi desktop Windows yang dirancang untuk mempermudah dan mempercepat alur kerja dokumentasi kantor atau acara. Aplikasi ini memungkinkan konversi massal foto RAW Canon (CR3/CR2) menjadi JPG berkualitas tinggi serta konversi video kamera menjadi format MP4 1080p FHD yang ramah WhatsApp dan media sosial lainnya tanpa kehilangan detail penting.

## Fitur Utama

1. **Konversi Foto RAW CR3/CR2 Massal**: Mengonversi file RAW Canon ke JPG tanpa mengurangi resolusi pixel asli secara default.
2. **Koreksi Gambar Otomatis (Autocorrect) Opsional**: Pilihan koreksi warna dan exposure (Natural, Bright, Vivid, Low Light, atau Off) yang konservatif untuk menghindari warna yang terlalu mencolok (overcooked).
3. **Konversi Video Massal**: Mengonversi video kamera (.mov, .mp4, .m4v, .mts) menjadi format MP4 (H.264 + AAC) yang sangat kompatibel dengan WhatsApp, Android, dan iPhone.
4. **Keamanan Data**: File asli tidak akan pernah ditimpa (overwritten) atau dihapus. Jika terjadi konflik nama file, aplikasi akan otomatis menambahkan akhiran unik seperti _1, _2, dll.
5. **Responsif**: Semua proses konversi berjalan di thread latar belakang (background worker thread), sehingga UI aplikasi tidak akan membeku (freeze).

---

## Cara Instalasi Python

Sebelum menjalankan aplikasi ini, pastikan Anda telah menginstal Python 3.11 atau versi yang lebih baru di sistem Windows Anda:

1. Unduh Python dari situs resmi: [python.org/downloads](https://www.python.org/downloads/).
2. Saat menjalankan installer, pastikan untuk mencentang opsi **"Add Python to PATH"** sebelum mengklik **Install Now**.

---

## Cara Menjalankan Aplikasi

Anda dapat menggunakan skrip otomatis un.ps1 atau menjalankannya secara manual dengan langkah-langkah berikut melalui PowerShell:

### Opsi 1: Menggunakan Skrip Otomatis
Jalankan skrip berikut di PowerShell untuk membuat virtual environment, menginstal dependensi, dan menjalankan aplikasi secara otomatis:
`powershell
.\run.ps1
`

### Opsi 2: Langkah Manual
1. Buka PowerShell di dalam direktori RAW2Share.
2. Buat Virtual Environment:
   `powershell
   python -m venv .venv
   `
3. Aktifkan Virtual Environment:
   `powershell
   .\.venv\Scripts\activate
   `
4. Instal dependensi yang diperlukan:
   `powershell
   pip install -r requirements.txt
   `
5. Jalankan aplikasi:
   `powershell
   python src/main.py
   `

---

## Cara Membangun Executable (.exe)

Untuk membuat aplikasi mandiri (standalone .exe) yang dapat dijalankan tanpa Python terinstal di komputer target:

Jalankan skrip build berikut di PowerShell:
`powershell
.\build_windows.ps1
`
Skrip ini akan secara otomatis:
1. Memastikan virtual environment dan dependensi terinstal.
2. Menjalankan seluruh pengujian unit (unit tests).
3. Membangun executable bernama RAW2Share.exe menggunakan PyInstaller.
4. Output executable akan berada di folder dist/RAW2Share/RAW2Share.exe.

---

## Rekomendasi Pengaturan

Untuk hasil terbaik, kami menyarankan pengaturan berikut:

* **Foto CR3 (RAW)**:
  * **JPG Quality**: Atur slider ke **98** (default) untuk rasio kompresi dan kualitas terbaik.
  * **Keep Original Resolution**: Pastikan opsi ini dicentang (ON) agar resolusi piksel asli foto tetap terjaga.
  * **Autocorrect Mode**: Pilih **Off** untuk warna asli dari kamera, atau **Natural** untuk koreksi exposure, kontras, saturasi, dan ketajaman secara ringan.
* **Video WhatsApp**:
  * **Preset**: Gunakan **WhatsApp Balanced 1080p** (default) untuk kompresi video yang optimal (CRF 23, AAC 160k, preset medium).
  * **Tips WhatsApp**: Meskipun resolusi tetap tinggi, JPG dan MP4 hasil konversi adalah format terkompresi. Untuk kualitas maksimal tanpa kompresi tambahan oleh WhatsApp saat pengiriman, kirimkan file sebagai **Dokumen/File** di WhatsApp, bukan melalui galeri foto biasa.

---

## Catatan Teknis
* Konversi RAW menggunakan pustaka awpy yang merupakan wrapper dari LibRaw.
* Konversi video menggunakan FFmpeg (melalui pustaka imageio-ffmpeg atau fallback ke executable sistem).
* Metadata EXIF pada foto akan disalin dari file RAW asli menggunakan alat pihak ketiga xiftool jika terdeteksi di komputer Anda. Jika tidak ditemukan, aplikasi akan tetap berjalan normal dengan menyalin metadata dasar saja.
