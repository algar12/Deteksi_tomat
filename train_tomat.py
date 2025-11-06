import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

base_dir = "/home/gopung/Desktop/deteksi_tomat/Dbset"
class_names = ["Matang", "Mentah", "Setengah Matang"]

def load_images_with_subfolders(base_path, class_names, target_size=(128, 128)):  # Ukuran lebih kecil
    images = []
    labels = []
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)
        img_count = 0
        
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    try:
                        img = tf.keras.utils.load_img(img_path, target_size=target_size)
                        img_array = tf.keras.utils.img_to_array(img)
                        images.append(img_array)
                        labels.append(label)
                        img_count += 1
                    except Exception as e:
                        print(f"Gagal membaca {img_path}: {e}")
        
        print(f"  ✓ {class_name}: {img_count} gambar")
    
    return np.array(images), np.array(labels)

print("📂 Membaca dataset...")
X, y = load_images_with_subfolders(base_dir, class_names)
print(f"\n✅ Total gambar terbaca: {len(X)}")

if len(X) == 0:
    print("❌ Tidak ada gambar yang terbaca! Cek struktur folder.")
    exit(1)

# Cek distribusi kelas
unique, counts = np.unique(y, return_counts=True)
print("\n📊 Distribusi Dataset:")
for i, count in zip(unique, counts):
    percentage = (count / len(y)) * 100
    print(f"   {class_names[i]}: {count} gambar ({percentage:.1f}%)")

# Normalisasi
X = X / 255.0

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Data Training: {len(X_train)} | Validasi: {len(X_val)}")

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(class_names))

# Data augmentation - MODERAT
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])

# MODEL SEDERHANA - Untuk dataset kecil
print("\n🏗️  Membangun model SEDERHANA...")
model = Sequential([
    data_augmentation,
    
    # Block 1 - Simple
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    
    # Block 2
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    
    # Block 3
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),
    
    # Classifier - Simple
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile dengan learning rate NORMAL
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Learning rate normal
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks - lebih permisif
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Lebih sabar
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        os.path.join(base_dir, "best_model_tomat.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n🚀 Memulai training...\n")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,  # Batch size lebih kecil
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

model.save(os.path.join(base_dir, "model_tomat_final.keras"))
print("\n✅ Model tersimpan sebagai 'model_tomat_final.keras'")

# Evaluasi
print("\n📈 Evaluasi Model:")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Prediksi per kelas
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

print("\n📊 Akurasi per Kelas:")
for i, class_name in enumerate(class_names):
    class_mask = y_val_classes == i
    if np.sum(class_mask) > 0:
        class_acc = np.mean(y_pred_classes[class_mask] == i) * 100
        print(f"   {class_name:20s}: {class_acc:.2f}%")

# Plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
plt.legend(loc='lower right')
plt.title('Akurasi Training dan Validasi', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
plt.legend(loc='upper right')
plt.title('Loss Training dan Validasi', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
print(f"\n📊 Grafik tersimpan sebagai 'training_history.png'")
plt.close()

# Tips
print("\n💡 Tips untuk Meningkatkan Akurasi:")
print("   1. Tambah lebih banyak data (minimal 100 gambar per kelas)")
print("   2. Pastikan gambar berkualitas baik (jelas, tidak blur)")
print("   3. Gunakan pencahayaan yang konsisten")
print("   4. Pertimbangkan transfer learning (MobileNet/ResNet)")