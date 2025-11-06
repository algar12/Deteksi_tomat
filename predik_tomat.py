import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# ===========================
# KONFIGURASI
# ===========================
base_dir = "/home/gopung/Desktop/deteksi_tomat/Dbset"
model_path = os.path.join(base_dir, "model_tomat_final.keras")
class_names = ["Matang", "Mentah", "Setengah Matang"]

# ===========================
# LOAD MODEL
# ===========================
print("📦 Memuat model...")
model = keras.models.load_model(model_path)
print("✅ Model berhasil dimuat!\n")

# ===========================
# FUNGSI PREPROCESSING
# ===========================
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load dan preprocess gambar untuk prediksi"""
    img = keras.utils.load_img(image_path, target_size=target_size)
    img_array = keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension
    return img_array, img

# ===========================
# PREDIKSI SINGLE IMAGE
# ===========================
def predict_single_image(image_path):
    img_array, original_img = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100

    print(f"🖼️  Gambar: {os.path.basename(image_path)}")
    print(f"🎯 Prediksi: {class_names[predicted_class]}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("\n📈 Probabilitas semua kelas:")
    for i, prob in enumerate(predictions[0]):
        print(f"   {class_names[i]}: {prob*100:.2f}%")

    # Visualisasi
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f"Prediksi: {class_names[predicted_class]}\nConfidence: {confidence:.1f}%", fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = ['green' if i == predicted_class else 'lightgray' for i in range(len(class_names))]
    bars = plt.bar(class_names, predictions[0]*100, color=colors)
    plt.ylabel('Probabilitas (%)')
    plt.title('Distribusi Probabilitas', fontsize=12, fontweight='bold')
    plt.ylim(0, 105)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.show()

    return predicted_class, confidence

# ===========================
# PREDIKSI BATCH
# ===========================
def predict_batch(folder_path, save_results=True):
    results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]

    print(f"🔍 Memproses {len(image_files)} gambar...\n")
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img_array, _ = load_and_preprocess_image(img_path)
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        results.append({
            'filename': img_file,
            'prediction': class_names[predicted_class],
            'confidence': confidence
        })
        print(f"✓ {img_file:30s} → {class_names[predicted_class]:15s} ({confidence:.1f}%)")

    if save_results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(folder_path, 'hasil_prediksi.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Hasil disimpan ke: {csv_path}")
    return results

# ===========================
# EVALUASI MODEL
# ===========================
def evaluate_model(base_dir, class_names):
    from sklearn.model_selection import train_test_split

    def load_images_with_subfolders(base_path, class_names, target_size=(128,128)):
        images, labels = [], []
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(base_path, class_name)
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.png','.jpg','.jpeg')):
                        try:
                            img = keras.utils.load_img(os.path.join(root,file), target_size=target_size)
                            images.append(keras.utils.img_to_array(img))
                            labels.append(label)
                        except: pass
        return np.array(images), np.array(labels)

    print("📂 Memuat dataset validasi...")
    X, y = load_images_with_subfolders(base_dir, class_names)
    X = X / 255.0
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✅ Dataset validasi: {len(X_val)} gambar\n")

    print("🔮 Melakukan prediksi...")
    y_pred = model.predict(X_val, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n📊 CLASSIFICATION REPORT")
    print(classification_report(y_val, y_pred_classes, target_names=class_names, digits=4))

    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label':'Jumlah Prediksi'})
    plt.title('Confusion Matrix - Deteksi Kematangan Tomat', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    cm_path = os.path.join(base_dir,'confusion_matrix.png')
    plt.savefig(cm_path,dpi=300,bbox_inches='tight')
    print(f"\n💾 Confusion matrix disimpan ke: {cm_path}")
    plt.show()

    print("\n🎯 AKURASI PER KELAS")
    for i, class_name in enumerate(class_names):
        class_accuracy = cm[i,i]/np.sum(cm[i,:])*100
        print(f"{class_name:20s}: {class_accuracy:.2f}%")
    overall_accuracy = np.trace(cm)/np.sum(cm)*100
    print(f"\n{'Overall Accuracy':20s}: {overall_accuracy:.2f}%")
    return cm, y_val, y_pred_classes

# ===========================
# MISCLASSIFIED IMAGES
# ===========================
def show_misclassified(base_dir, class_names, num_samples=9):
    from sklearn.model_selection import train_test_split

    def load_images(base_path, class_names, target_size=(128,128)):
        images, labels, filenames = [], [], []
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(base_path,class_name)
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.png','.jpg','.jpeg')):
                        try:
                            img = keras.utils.load_img(os.path.join(root,file), target_size=target_size)
                            images.append(keras.utils.img_to_array(img))
                            labels.append(label)
                            filenames.append(os.path.join(root,file))
                        except: pass
        return np.array(images), np.array(labels), filenames

    X, y, filenames = load_images(base_dir, class_names)
    X = X / 255.0
    _, X_val, _, y_val, _, filenames_val = train_test_split(X, y, filenames, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    mis_idx = np.where(y_pred_classes!=y_val)[0]
    if len(mis_idx)==0:
        print("🎉 Tidak ada gambar yang salah diprediksi!")
        return
    print(f"❌ Ditemukan {len(mis_idx)} gambar yang salah diprediksi")

    num_samples = min(num_samples,len(mis_idx))
    sample_idx = np.random.choice(mis_idx, num_samples, replace=False)
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples/rows))
    plt.figure(figsize=(cols*4, rows*4))

    for i, idx in enumerate(sample_idx):
        plt.subplot(rows,cols,i+1)
        img = keras.utils.load_img(filenames_val[idx], target_size=(128,128))
        plt.imshow(img)
        true_label = class_names[y_val[idx]]
        pred_label = class_names[y_pred_classes[idx]]
        confidence = y_pred[idx][y_pred_classes[idx]]*100
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", fontsize=10,color='red',fontweight='bold')
        plt.axis('off')
    plt.suptitle('Contoh Prediksi yang Salah', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# ===========================
# MENU UTAMA
# ===========================
def main():
    print("="*60)
    print(" 🍅 SISTEM DETEKSI TINGKAT KEMATANGAN TOMAT 🍅")
    print("="*60)
    while True:
        print("\nPilih mode operasi:")
        print("1. Prediksi gambar tunggal")
        print("2. Prediksi batch (folder)")
        print("3. Evaluasi model lengkap")
        print("4. Tampilkan prediksi yang salah")
        print("5. Keluar")
        choice = input("\nPilihan Anda (1-5): ")

        if choice=="1":
            image_path = input("\n📁 Masukkan path gambar: ")
            if os.path.exists(image_path):
                predict_single_image(image_path)
            else:
                print("❌ File tidak ditemukan!")
        elif choice=="2":
            folder_path = input("\n📁 Masukkan path folder: ")
            if os.path.exists(folder_path):
                predict_batch(folder_path)
            else:
                print("❌ Folder tidak ditemukan!")
        elif choice=="3":
            evaluate_model(base_dir, class_names)
        elif choice=="4":
            num_samples = int(input("\n🔢 Berapa gambar yang ingin ditampilkan? (default: 9): ") or "9")
            show_misclassified(base_dir, class_names, num_samples)
        elif choice=="5":
            print("\n👋 Terima kasih!")
            break
        else:
            print("❌ Pilihan tidak valid!")

if __name__=="__main__":
    main()
