import streamlit as st
import cv2
import numpy as np
import os
import pytesseract
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

if os.name == "nt":  # Windows uniquement
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="ALPR - Lecture Automatique de Plaques",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    .stDownloadButton button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1>🚗 ALPR - Système de Lecture Automatique de Plaques</h1>
        <p style='font-size: 1.1rem; color: #6b7280;'>
            Détection intelligente, extraction OCR et vérification par réseau de neurones
        </p>
    </div>
""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/license-plate.png", width=80)
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("### 📊 Informations")
    st.info("""
    **Comment ça marche ?**
    1. 📤 Téléchargez une image
    2. 🔍 Détection automatique
    3. 📝 Extraction OCR
    4. 🧠 Vérification CNN
    """)
    
    
    
    st.markdown("---")
    st.caption("Version 2.0 - 2024")

# Path Tesseract (à adapter selon ton PC)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Charger le modèle CNN sauvegardé
model = load_model("cnn_mnist_model.h5")

# Charger Haar Cascade pour la détection de plaque
CASCADE_PATH = "haar_cascades/haarcascade_russian_plate_number.xml"
carplate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ================== FONCTIONS ==================

def detect_plate(image):
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    plate_img = image_rgb.copy()
    plates = carplate_cascade.detectMultiScale(plate_img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in plates:
        cv2.rectangle(plate_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        plate_img_crop = image_rgb[y+15:y+h-10, x+15:x+w-20]
        return plate_img, plate_img_crop
    return plate_img, None


def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 3)
    return blur


def ocr_text(plate_img):
    text = pytesseract.image_to_string(
        plate_img,
        config="--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    return text.strip()


def cnn_predict_digits_exact(plate_img, model):
    gray = cv2.cvtColor(np.array(plate_img), cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    h_img, _ = thresh.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > h_img * 0.3:
            candidates.append((x, y, w, h))

    candidates.sort(key=lambda x: x[0])
    predicted_text = ""

    for (x, y, w, h) in candidates:
        roi = thresh[y:y+h, x:x+w]
        h_roi, w_roi = roi.shape
        scale = 20.0 / max(h_roi, w_roi)
        new_h, new_w = int(h_roi * scale), int(w_roi * scale)
        resized = cv2.resize(roi, (new_w, new_h))
        padded = np.zeros((28,28), dtype=np.uint8)
        pad_y, pad_x = (28-new_h)//2, (28-new_w)//2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        input_data = padded.reshape(1,28,28,1)/255.0
        pred = model.predict(input_data, verbose=0)
        class_id = np.argmax(pred)
        predicted_text += str(class_id)

    return predicted_text


# ================== INTERFACE ==================
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Téléchargez une image de voiture", 
    type=["png","jpg","jpeg"],
    help="Formats acceptés: PNG, JPG, JPEG"
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Affichage de l'image originale en haut
    st.markdown("### 📸 Image Originale")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, use_column_width=True)
    
    st.markdown("---")
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔎 Détection de la plaque en cours...")
    progress_bar.progress(25)
    
    annotated, plate_crop = detect_plate(image)
    progress_bar.progress(50)
    
    if plate_crop is None:
        st.error("❌ Aucune plaque détectée. Veuillez essayer avec une autre image.")
        progress_bar.empty()
        status_text.empty()
    else:
        status_text.text("✅ Plaque détectée avec succès!")
        progress_bar.progress(100)
        
        # Affichage de la détection
        st.markdown("### 🎯 Résultat de la Détection")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Image avec annotation**")
            st.image(annotated, use_column_width=True)
        
        with col2:
            st.markdown("**Plaque extraite**")
            st.image(plate_crop, use_column_width=True)
            
            # Bouton de téléchargement
            buf = cv2.imencode('.png', cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR))[1].tobytes()
            st.download_button(
                "⬇️ Télécharger la plaque",
                data=buf,
                file_name="plate_detected.png",
                mime="image/png"
            )
        
        st.markdown("---")
        
        # Traitement OCR et CNN
        plate_processed = preprocess_for_ocr(plate_crop)
        
        st.markdown("### 🔍 Analyse et Reconnaissance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("📝 Extraction OCR..."):
                ocr_result = ocr_text(plate_processed)
            st.markdown(f"""
                <div class='result-box'>
                    <div style='font-size: 1rem; opacity: 0.9;'>OCR Tesseract</div>
                    <div style='font-size: 2rem; margin-top: 0.5rem;'>{ocr_result if ocr_result else "---"}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            with st.spinner("🧠 Vérification CNN..."):
                cnn_result = cnn_predict_digits_exact(plate_crop, model)
            st.markdown(f"""
                <div class='result-box' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                    <div style='font-size: 1rem; opacity: 0.9;'>CNN MNIST</div>
                    <div style='font-size: 2rem; margin-top: 0.5rem;'>{cnn_result if cnn_result else "---"}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparaison finale
        st.markdown("### 📊 Résultat Final")
        
        if ocr_result and cnn_result:
            if ocr_result == cnn_result:
                st.success("✅ **VALIDATION RÉUSSIE** - Les deux méthodes correspondent parfaitement!")
                st.balloons()
            else:
                st.warning("⚠️ **DIVERGENCE DÉTECTÉE** - Les résultats diffèrent entre OCR et CNN")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("OCR", ocr_result)
                with col2:
                    st.metric("CNN", cnn_result)
                with col3:
                    similarity = sum(a == b for a, b in zip(ocr_result, cnn_result)) / max(len(ocr_result), len(cnn_result)) * 100
                    st.metric("Similarité", f"{similarity:.0f}%")
        
        progress_bar.empty()
        status_text.empty()

else:
    # Message d'accueil
    st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white;'>
            <h2>👆 Commencez par télécharger une image</h2>
            <p style='font-size: 1.1rem; margin-top: 1rem;'>
                Notre système analysera automatiquement la plaque d'immatriculation
            </p>
        </div>
    """, unsafe_allow_html=True)