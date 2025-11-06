import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import plotly.graph_objects as go
import os

# ==============================
# CONFIGURASI HALAMAN & STYLE
# ==============================
st.set_page_config(
    page_title="Deteksi Kematangan Tomat",
    page_icon="🍅",
    layout="wide"
)

st.markdown("""
<style>
.main {padding: 0rem 1rem;}
.stButton>button {
    width: 100%;
    background-color: #FF4444;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
}
.stButton>button:hover {background-color: #CC0000;}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# KONSTANTA
# ==============================
BASE_DIR = "/home/gopung/Desktop/deteksi_tomat/Dbset"
MODEL_PATH = os.path.join(BASE_DIR, "model_tomat_final.keras")
CLASS_NAMES = ["Matang", "Mentah", "Setengah Matang"]
COLORS = {
    "Matang": "#FF4444",
    "Mentah": "#44FF44",
    "Setengah Matang": "#FFAA44"
}

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# ==============================
# PREPROCESSING
# ==============================
def preprocess_image(image):
    img = image.resize((128, 128))  # Sesuai training model
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    return predicted_class, confidence, predictions[0]

# ==============================
# HEADER
# ==============================
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>🍅 Deteksi Tingkat Kematangan Tomat</h1>
    <p style='color: white; margin-top: 0.5rem;'>Powered by Deep Learning & Computer Vision</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("📊 Info Model")
    st.info(f"""
**Kelas Deteksi:**  
- 🔴 Matang  
- 🟢 Mentah  
- 🟠 Setengah Matang  

**Arsitektur:** CNN  
**Input Size:** 128x128 px  
**Model:** Keras/TensorFlow
""")
    st.header("ℹ️ Cara Penggunaan")
    st.write("""
1. Upload gambar tomat  
2. Klik tombol 'Prediksi'  
3. Lihat hasil deteksi
""")
    st.header("📝 Catatan")
    st.warning("""
Untuk hasil terbaik:  
- Gunakan gambar yang jelas  
- Pastikan tomat terlihat dengan baik  
- Pencahayaan cukup
""")

# ==============================
# MAIN CONTENT
# ==============================
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar tomat",
        type=['jpg','jpeg','png'],
        help="Format: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
        
        if st.button("🔮 PREDIKSI"):
            with st.spinner('Memproses prediksi...'):
                predicted_class, confidence, probabilities = predict_image(model, image)
                predicted_label = CLASS_NAMES[predicted_class]
                
                # Simpan hasil ke session_state
                st.session_state['predicted_class'] = predicted_class
                st.session_state['predicted_label'] = predicted_label
                st.session_state['confidence'] = confidence
                st.session_state['probabilities'] = probabilities

with col2:
    st.subheader("🎯 Hasil Prediksi")
    if 'predicted_label' in st.session_state:
        predicted_label = st.session_state['predicted_label']
        confidence = st.session_state['confidence']
        probabilities = st.session_state['probabilities']
        
        # Result box
        color = COLORS[predicted_label]
        st.markdown(f"""
<div style='background-color: {color}; padding: 30px; border-radius: 15px; 
            text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h2 style='color: white; margin: 0; font-size: 2rem;'>{predicted_label}</h2>
    <p style='color: white; margin: 10px 0 0 0; font-size: 1.2rem;'>Confidence: {confidence:.2f}%</p>
</div>
""", unsafe_allow_html=True)
        
        # Probability chart
        st.subheader("📊 Distribusi Probabilitas")
        fig = go.Figure(data=[
            go.Bar(
                x=CLASS_NAMES,
                y=probabilities*100,
                marker_color=[COLORS[name] for name in CLASS_NAMES],
                text=[f'{p*100:.1f}%' for p in probabilities],
                textposition='outside',
            )
        ])
        fig.update_layout(
            yaxis_title="Probabilitas (%)",
            xaxis_title="Kelas",
            height=400,
            showlegend=False,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed probabilities
        st.subheader("📈 Detail Probabilitas")
        for class_name, prob in zip(CLASS_NAMES, probabilities):
            col_a, col_b = st.columns([3,1])
            with col_a:
                st.progress(float(prob))
            with col_b:
                st.markdown(f"<span style='color:{COLORS[class_name]}; font-weight:bold'>{prob*100:.2f}%</span>", unsafe_allow_html=True)
        
        # Interpretation
        st.subheader("💡 Interpretasi")
        if confidence > 90:
            st.success("✅ Model sangat yakin dengan prediksi ini!")
        elif confidence > 70:
            st.info("ℹ️ Model cukup yakin dengan prediksi ini.")
        else:
            st.warning("⚠️ Model kurang yakin. Coba gambar yang lebih jelas.")
    else:
        st.info("👆 Upload gambar dan klik tombol prediksi untuk melihat hasil")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Developed with ❤️ using TensorFlow & Streamlit</p>
    <p style='font-size: 0.9rem;'>© 2024 Tomat Detection System</p>
</div>
""", unsafe_allow_html=True)
