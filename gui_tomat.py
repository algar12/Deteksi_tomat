import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from tensorflow import keras

class TomatDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🍅 Deteksi Tingkat Kematangan Tomat")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # =======================
        # KONFIGURASI MODEL
        # =======================
        self.base_dir = "/home/gopung/Desktop/deteksi_tomat/Dbset"
        self.model_path = os.path.join(self.base_dir, "model_tomat_final.keras")
        self.class_names = ["Matang", "Mentah", "Setengah Matang"]
        self.colors = {
            "Matang": "#FF4444",
            "Mentah": "#44FF44", 
            "Setengah Matang": "#FFAA44"
        }
        
        self.model = None
        self.current_image_path = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        tk.Label(header_frame, text="🍅 DETEKSI TINGKAT KEMATANGAN TOMAT", 
                 font=('Arial', 20, 'bold'), bg='#2c3e50', fg='white').pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Image display
        left_panel = tk.Frame(main_container, bg='white', relief='ridge', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        self.image_canvas = tk.Label(left_panel, bg='white', text="Gambar akan ditampilkan di sini", fg='gray', font=('Arial',12))
        self.image_canvas.pack(pady=20, padx=10)
        
        # Buttons
        button_frame = tk.Frame(left_panel, bg='white')
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="📁 Pilih Gambar", font=('Arial',12,'bold'), bg='#3498db', fg='white',
                  padx=20, pady=10, relief='flat', cursor='hand2', command=self.select_image).pack(side='left', padx=5)
        tk.Button(button_frame, text="🔮 Prediksi", font=('Arial',12,'bold'), bg='#2ecc71', fg='white',
                  padx=20, pady=10, relief='flat', cursor='hand2', command=self.predict).pack(side='left', padx=5)
        tk.Button(button_frame, text="🗑️ Clear", font=('Arial',12,'bold'), bg='#e74c3c', fg='white',
                  padx=20, pady=10, relief='flat', cursor='hand2', command=self.clear).pack(side='left', padx=5)
        
        # Right panel - Results
        right_panel = tk.Frame(main_container, bg='white', relief='ridge', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        tk.Label(right_panel, text="📊 HASIL PREDIKSI", font=('Arial',16,'bold'), bg='white', fg='#2c3e50').pack(pady=20)
        self.result_frame = tk.Frame(right_panel, bg='white')
        self.result_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(side='bottom', fill='x')
        self.status_label = tk.Label(status_frame, text="Status: Siap", font=('Arial',10), bg='#34495e', fg='white', anchor='w')
        self.status_label.pack(side='left', padx=10)
    
    def load_model(self):
        try:
            self.status_label.config(text="Status: Memuat model...")
            self.root.update()
            self.model = keras.models.load_model(self.model_path)
            self.status_label.config(text="Status: Model berhasil dimuat ✓")
            messagebox.showinfo("Sukses", "Model berhasil dimuat!")
        except Exception as e:
            self.status_label.config(text="Status: Gagal memuat model ✗")
            messagebox.showerror("Error", f"Gagal memuat model:\n{str(e)}")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Tomat",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.status_label.config(text=f"Status: Gambar dimuat - {os.path.basename(file_path)}")
    
    def display_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((400,400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_canvas.config(image=photo, text='')
        self.image_canvas.image = photo
    
    def predict(self):
        if not self.current_image_path:
            messagebox.showwarning("Peringatan", "Silakan pilih gambar terlebih dahulu!")
            return
        if not self.model:
            messagebox.showerror("Error", "Model belum dimuat!")
            return
        try:
            self.status_label.config(text="Status: Memproses prediksi...")
            self.root.update()
            
            img = keras.utils.load_img(self.current_image_path, target_size=(128,128))
            img_array = keras.utils.img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            
            self.display_results(predicted_class, predictions[0])
            confidence = predictions[0][predicted_class]*100
            self.status_label.config(text=f"Status: Prediksi selesai - {self.class_names[predicted_class]} ({confidence:.1f}%)")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan prediksi:\n{str(e)}")
            self.status_label.config(text="Status: Prediksi gagal ✗")
    
    def display_results(self, predicted_class, probabilities):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        predicted_label = self.class_names[predicted_class]
        confidence = probabilities[predicted_class]*100
        color = self.colors[predicted_label]
        
        # Main prediction
        result_box = tk.Frame(self.result_frame, bg=color, relief='solid', bd=3)
        result_box.pack(pady=10, padx=10, fill='x')
        tk.Label(result_box, text="HASIL DETEKSI", font=('Arial',12,'bold'), bg=color, fg='white').pack(pady=(10,5))
        tk.Label(result_box, text=predicted_label, font=('Arial',24,'bold'), bg=color, fg='white').pack(pady=5)
        tk.Label(result_box, text=f"Confidence: {confidence:.2f}%", font=('Arial',14), bg=color, fg='white').pack(pady=(5,10))
        
        # Probability bars
        prob_frame = tk.Frame(self.result_frame, bg='white')
        prob_frame.pack(pady=20, padx=10, fill='both', expand=True)
        tk.Label(prob_frame, text="Probabilitas Semua Kelas:", font=('Arial',12,'bold'), bg='white').pack(pady=10)
        
        for class_name, prob in zip(self.class_names, probabilities):
            class_frame = tk.Frame(prob_frame, bg='white')
            class_frame.pack(pady=5, padx=20, fill='x')
            tk.Label(class_frame, text=f"{class_name}:", font=('Arial',11), bg='white', width=15, anchor='w').pack(side='left')
            progress = ttk.Progressbar(class_frame, length=200, mode='determinate', value=prob*100)
            progress.pack(side='left', padx=10)
            tk.Label(class_frame, text=f"{prob*100:.1f}%", font=('Arial',11,'bold'), bg='white', fg=self.colors[class_name]).pack(side='left')
    
    def clear(self):
        self.current_image_path = None
        self.image_canvas.config(image='', text="Gambar akan ditampilkan di sini", fg='gray')
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        self.status_label.config(text="Status: Siap")

def main():
    root = tk.Tk()
    app = TomatDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
