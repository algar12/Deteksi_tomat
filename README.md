# 🍅 Sistem Deteksi Tingkat Kematangan Tomat

Sistem deteksi otomatis untuk mengklasifikasikan tingkat kematangan tomat menggunakan Deep Learning (Convolutional Neural Network).

## 📋 Daftar Isi
- [Fitur](#fitur)
- [Instalasi](#instalasi)
- [Struktur Dataset](#struktur-dataset)
- [Cara Penggunaan](#cara-penggunaan)
- [Arsitektur Model](#arsitektur-model)
- [Hasil Training](#hasil-training)
- [File-File Penting](#file-file-penting)

## ✨ Fitur

- **3 Kelas Deteksi:**
  - 🔴 Matang
  - 🟢 Mentah
  - 🟠 Setengah Matang

- **Multiple Interface:**
  - Command Line Interface (CLI)
  - Graphical User Interface (GUI) dengan Tkinter
  - Web Application dengan Streamlit

- **Fitur Lengkap:**
  - Data augmentation
  - Batch normalization
  - Dropout regularization
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing

## 🚀 Instalasi

### 1. Clone atau Download Repository

```bash
cd ~/Desktop/deteksi_tomat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
seaborn>=0.12.0
pandas>=2.0.0
streamlit>=1.28.0
plotly>=5.17.0
```

Atau install manual:
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow seaborn pandas streamlit plotly
```

## 📁 Struktur Dataset

```
Dbset/
├── Matang/
│   ├── tomat1.jpg
│   ├── tomat2.jpg
│   └── ...
├── Mentah/
│   ├── tomat1.jpg
│   ├── tomat2.jpg
│   └── ...
└── Setengah Matang/
    ├── tomat1.jpg
    ├── tomat2.jpg
    └── ...
```

## 📖 Cara Penggunaan

### 1. Training Model

```bash
python train_tomat.py
```

**Output:**
- `best_model_tomat.keras` - Model terbaik berdasarkan validation accuracy
- `model_tomat_final.keras` - Model di epoch terakhir
- `training_history.png` - Grafik akurasi dan loss

### 2. Prediksi dengan CLI

```bash
python predict_tomat.py
```

**Menu yang tersedia:**
1. Prediksi gambar tunggal
2. Prediksi batch (folder)
3. Evaluasi model lengkap
4. Tampilkan prediksi yang salah
5. Keluar

### 3. GUI Application (Tkinter)

```bash
python gui_tomat.py
```

**Fitur GUI:**
- Upload gambar dengan file dialog
- Preview gambar yang diupload
- Prediksi real-time
- Visualisasi hasil dengan progress bar
- Color-coded results

### 4. Web Application (Streamlit)

```bash
streamlit run app_tomat.py
```

Kemudian buka browser di `http://localhost:8501`

**Fitur Web App:**
- Modern web interface
- Drag & drop file upload
- Interactive charts dengan Plotly
- Responsive design
- Real-time prediction

## 🏗️ Arsitektur Model

```
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
data_augmentation           (None, 180, 180, 3)       0
_________________________________________________________________
conv2d_1 (Conv2D)           (None, 180, 180, 32)      896
batch_normalization_1       (None, 180, 180, 32)      128
conv2d_2 (Conv2D)           (None, 180, 180, 32)      9,248
max_pooling2d_1             (None, 90, 90, 32)        0
dropout_1 (Dropout)         (None, 90, 90, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)           (None, 90, 90, 64)        18,496
batch_normalization_2       (None, 90, 90, 64)        256
conv2d_4 (Conv2D)           (None, 90, 90, 64)        36,928
max_pooling2d_2             (None, 45, 45, 64)        0
dropout_2 (Dropout)         (None, 45, 45, 64)        0
_________________________________________________________________
conv2d_5 (Conv2D)           (None, 45, 45, 128)       73,856
batch_normalization_3       (None, 45, 45, 128)       512
conv2d_6 (Conv2D)           (None, 45, 45, 128)       147,584
max_pooling2d_3             (None, 22, 22, 128)       0
dropout_3 (Dropout)         (None, 22, 22, 128)       0
_________________________________________________________________
flatten                     (None, 61,952)            0
dense_1 (Dense)             (None, 256)               15,859,968
batch_normalization_4       (None, 256)               1,024
dropout_4 (Dropout)         (None, 256)               0
dense_2 (Dense)             (None, 128)               32,896
dropout_5 (Dropout)         (None, 128)               0
dense_3 (Dense)             (None, 3)                 387
=================================================================
Total params: 16,182,179
Trainable params: 16,181,219
Non-trainable params: 960
```

## 📊 Hasil Training

### Hyperparameters:
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 50 (dengan early stopping)
- **Batch Size:** 32
- **Train/Val Split:** 80/20
- **Data Augmentation:** Horizontal/vertical flip, rotation, zoom, brightness

### Callbacks:
- **EarlyStopping:** Patience 7 epochs
- **ReduceLROnPlateau:** Reduce LR ketika val_loss plateau
- **ModelCheckpoint:** Save best model berdasarkan val_accuracy

### Contoh Output:
```
📂 Membaca dataset...
  ✓ Matang: 150 gambar
  ✓ Mentah: 140 gambar
  ✓ Setengah Matang: 145 gambar

✅ Total gambar terbaca: 435
   Matang: 150 gambar
   Mentah: 140 gambar
   Setengah Matang: 145 gambar

📊 Data Training: 348 | Validasi: 87

🚀 Memulai training...

Epoch 1/50: loss: 0.8234 - accuracy: 0.6523 - val_loss: 0.5432 - val_accuracy: 0.7816
Epoch 2/50: loss: 0.4567 - accuracy: 0.8234 - val_loss: 0.3891 - val_accuracy: 0.8621
...
Epoch 25/50: loss: 0.1234 - accuracy: 0.9543 - val_loss: 0.2156 - val_accuracy: 0.9195

✅ Model tersimpan sebagai 'model_tomat_final.keras'
```

## 📁 File-File Penting

| File | Deskripsi |
|------|-----------|
| `train_tomat.py` | Script untuk training model |
| `predict_tomat.py` | CLI untuk prediksi dan evaluasi |
| `gui_tomat.py` | GUI application dengan Tkinter |
| `app_tomat.py` | Web application dengan Streamlit |
| `best_model_tomat.keras` | Model terbaik (auto-saved) |
| `model_tomat_final.keras` | Model final setelah training |
| `training_history.png` | Grafik training history |
| `confusion_matrix.png` | Confusion matrix hasil evaluasi |

## 🎯 Tips Penggunaan

### Untuk Hasil Prediksi Terbaik:
1. Gunakan gambar dengan pencahayaan yang baik
2. Tomat harus terlihat jelas (tidak blur)
3. Hindari gambar dengan background yang terlalu ramai
4. Resolusi minimal 180x180 pixels

### Troubleshooting:

**Error: Model tidak ditemukan**
```bash
# Pastikan path model benar
MODEL_PATH = "/home/gopung/Desktop/deteksi_tomat/Dbset/best_model_tomat.keras"
```

**Error: Out of Memory saat training**
```python
# Kurangi batch size
batch_size = 16  # dari 32
```

**Akurasi rendah**
- Tambah data training
- Tingkatkan epochs
- Adjust learning rate
- Coba data augmentation yang lebih agresif

## 📈 Evaluasi Model

### Classification Report:
```python
python predict_tomat.py
# Pilih option 3: Evaluasi model lengkap
```

### Confusion Matrix:
Akan menghasilkan heatmap yang menunjukkan:
- True Positives
- False Positives
- True Negatives
- False Negatives

### Metrics:
- **Precision:** Seberapa akurat prediksi positif
- **Recall:** Seberapa banyak positif yang terdeteksi
- **F1-Score:** Harmonic mean dari precision dan recall
- **Accuracy:** Overall accuracy

## 🔄 Update Model

Untuk melatih ulang model dengan data baru:

1. Tambahkan data ke folder yang sesuai
2. Jalankan training ulang:
```bash
python train_tomat.py
```
3. Model baru akan menimpa yang lama

## 📝 Lisensi

MIT License - Bebas digunakan untuk keperluan pendidikan dan komersial.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- TensorFlow/Keras team
- OpenCV community
- Dataset contributors

## 📞 Support

Jika ada pertanyaan atau bug, silakan buat issue di repository atau hubungi:
- Email: your.email@example.com
- Telegram: @yourusername

---

**Happy Detecting! 🍅**