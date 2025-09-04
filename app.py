import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from twilio.rest import Client
from datetime import datetime

# --------------------------
# Streamlit Page Setup
# --------------------------
st.set_page_config(
    page_title="Tomato Guardian • Current & Future Disease Prediction",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# THEME & CUSTOM CSS
# --------------------------
CSS = """
<style>
/* Gradient background */
.stApp {
  background: radial-gradient(1200px 600px at 10% -20%, #ffe8ec 0%, rgba(255,255,255,0) 60%),
              linear-gradient(135deg, #fef6f0 0%, #eef7ff 45%, #f3f0ff 100%);
  background-attachment: fixed;
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
/* Hero */
.hero {padding: 34px 28px; border-radius: 24px; background: linear-gradient(135deg, rgba(255,99,132,0.16), rgba(54, 162, 235, 0.16));
border: 1px solid rgba(255, 99, 132, 0.25); box-shadow: 0 15px 40px rgba(0,0,0,0.06);}
.hero h1 {font-weight: 800; margin: 0 0 6px 0; font-size: 2rem;}
.hero p {margin: 0; font-size: 0.98rem; opacity: 0.9;}
.badge {display: inline-block; padding: 6px 12px; border-radius: 999px; background: rgba(255, 99, 132, 0.12);
color: #b30038; border: 1px solid rgba(255, 99, 132, 0.26); font-weight: 600; font-size: 12px; margin-bottom: 10px;}
/* Cards */
.card {background: rgba(255,255,255,0.65); backdrop-filter: blur(10px);
border: 1px solid rgba(120,120,180,0.2); border-radius: 20px; padding: 18px;}
/* Footer */
.footer {opacity: 0.7; text-align:center; font-size:12px; margin-top: 28px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------
# Sidebar (Quick Guide)
# --------------------------
with st.sidebar:
    st.markdown("### 🍅 Tomato Guardian")
    st.markdown(
        """
**How it works**
1. Upload or capture a leaf photo → CNN classifies current disease  
2. Pull IoT climate (ThingSpeak) → Forecast future risk  
3. Auto-send SMS guidance to farmer  
        """
    )
    st.divider()
    st.caption("⚠️ Keys are hardcoded (for local test). In production, use st.secrets.")

# --------------------------
# Load Models
# --------------------------
@st.cache_resource(show_spinner=False)
def _load_artifacts():
    leaf_model = load_model("tomato_disease_leaf_cnn.h5")
    climate_model = load_model("tomato_disease_cnn.h5")
    scaler = joblib.load("scaler_cnn.pkl")
    le = joblib.load("label_encoder_cnn.pkl")
    return leaf_model, climate_model, scaler, le

leaf_model, climate_model, scaler, le = _load_artifacts()

# --------------------------
# IoT Cloud Config (ThingSpeak)
# --------------------------
API_KEY = "8U80HP8TTJ3SAQZ9"
CHANNEL_ID = "3054191"
url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json?api_key={API_KEY}"

# --------------------------
# Twilio Config
# --------------------------
ACCOUNT_SID = "AC223061301153005b7f48c59355749163"
AUTH_TOKEN  = "48ac078177fed9020fd859d17f29d648"
TWILIO_PHONE = "+12675542248"
FARMER_PHONE = "+918838712068"

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# --------------------------
# Disease Suggestions
# --------------------------
disease_suggestions = {
    "Bacterial_spot": "Use copper sprays; avoid overhead irrigation.",
    "Early_blight": "Apply fungicides; remove infected debris.",
    "Late_blight": "Destroy infected plants; grow resistant varieties.",
    "Leaf_Mold": "Improve ventilation; reduce humidity.",
    "powdery_mildew": "Use neem oil or sulfur spray.",
    "Spider_mites": "Use neem oil or miticides.",
    "Target_Spot": "Remove diseased leaves; apply fungicides.",
    "Tomato_mosaic_virus": "Plant resistant varieties; disinfect tools.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies; use resistant hybrids.",
    "healthy": "No disease detected. Keep monitoring your crop."
}

leaf_classes = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Spider_mites",
    "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
    "healthy", "powdery_mildew"
]

# --------------------------
# Helpers
# --------------------------
def preprocess_leaf(img: Image.Image):
    img = img.resize((224, 224))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_leaf(img: Image.Image):
    arr = preprocess_leaf(img)
    preds = leaf_model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    return leaf_classes[idx], float(preds[0][idx]) * 100.0

def predict_climate(temp, hum, soil):
    X_input = np.array([[temp, hum, soil]])
    X_scaled = scaler.transform(X_input)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    pred_probs = climate_model.predict(X_reshaped, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)
    return le.inverse_transform(pred_class)[0]

def send_sms(msg):
    try:
        client.messages.create(
            body=msg, from_=TWILIO_PHONE, to=FARMER_PHONE
        )
        return True
    except Exception as e:
        return str(e)

# --------------------------
# HERO
# --------------------------
colA, colB = st.columns([1.2, 1])
with colA:
    st.markdown(
        f"""
        <div class="hero">
          <div class="badge">AI + IoT + Alerts</div>
          <h1>Tomato Guardian</h1>
          <p>Smart plant health assistant that detects current leaf diseases, predicts future risks using climate data,
          and sends instant SMS guidance to the farmer.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with colB:
    now = datetime.now().strftime("%d %b %Y • %I:%M %p")
    st.markdown(f"""
    <div class="card">
      <b>📡 System Status</b><br><br>
      ✅ Models Loaded<br>
      ✅ Twilio Ready<br>
      ✅ ThingSpeak Connected<br><br>
      <span style="font-size:12px;opacity:0.7;">Last refreshed: {now}</span>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# Current Disease Detection
# --------------------------
st.header("📸 Current Disease Detection (Leaf Image)")
option = st.radio("Choose input method:", ["Upload Photo", "Use Camera"])
img = None
sms_text = ""

if option == "Upload Photo":
    file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if file: img = Image.open(file).convert("RGB")
else:
    file = st.camera_input("Take a leaf photo")
    if file: img = Image.open(file).convert("RGB")

if img:
    st.image(img, caption="Leaf Image", use_container_width=True)
    label, conf = predict_leaf(img)
    if label == "healthy":
        st.success(f"✅ Leaf looks {label} ({conf:.2f}%)")
        sms_text += f"Current: Plant is Healthy ({conf:.1f}%)\n"
    else:
        st.error(f"⚠ Disease Detected: {label} ({conf:.2f}%)")
        st.info(f"💡 Suggestion: {disease_suggestions.get(label)}")
        sms_text += f"Current Disease: {label} ({conf:.1f}%)\nSuggestion: {disease_suggestions.get(label)}\n"
    st.session_state["sms_current"] = sms_text

# --------------------------
# Future Prediction
# --------------------------
st.header("🌤 Future Disease Prediction (Climate Data)")
days = st.multiselect("📅 Select number of future days:", options=[1,2,3,4,5,6,7])

if st.button("🚀 Predict Future Diseases"):
    if not days:
        st.warning("⚠ Please select at least one day.")
        st.stop()
    try:
        response = requests.get(url, timeout=5).json()
        temp = float(response.get("field1") or 0)
        hum = float(response.get("field2") or 0)
        soil = float(response.get("field3") or 0)
    except Exception as e:
        st.error(f"⚠ Failed to fetch IoT data: {e}")
        st.stop()

    if temp == 0 and hum == 0 and soil == 0:
        st.warning("❌ No valid IoT data available.")
        st.stop()

    st.write(f"🌡 Temperature: {temp} °C")
    st.write(f"💧 Humidity: {hum} %")
    st.write(f"🌱 Soil Moisture: {soil} %")

    forecast_data = np.array([[temp + d*1.5, hum - d*2, soil + d*1.0] for d in days])
    diseases = [predict_climate(*row) for row in forecast_data]

    st.subheader("📊 Future Predictions:")
    sms_text = st.session_state.get("sms_current", "") + "Future Predictions:\n"
    for d, disease in zip(days, diseases):
        if disease.lower() == "healthy":
            st.success(f"Day {d} → ✅ {disease}")
        else:
            st.error(f"Day {d} → ⚠ {disease}")
            st.info(f"💡 Suggestion: {disease_suggestions.get(disease)}")
        sms_text += f"Day {d}: {disease}\n"

    sms_status = send_sms("🌱 Tomato Disease Report\n" + sms_text)
    if sms_status is True:
        st.success("📩 SMS sent successfully with both current & future predictions!")
    else:
        st.error(f"❌ SMS failed: {sms_status}")

# --------------------------
# Footer
# --------------------------
st.markdown(
    """
    <div class="footer">
      Built with ❤️ using Streamlit, TensorFlow, ThingSpeak & Twilio
    </div>
    """,
    unsafe_allow_html=True
)
