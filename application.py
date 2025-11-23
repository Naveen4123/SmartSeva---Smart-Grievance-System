import streamlit as st
import os
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

st.set_page_config(page_title="SmartSeva AI", layout="wide")

# =====================================================================================
#                         GOOGLE DRIVE MODEL DOWNLOADING + CACHING
# =====================================================================================

MODEL_PATH = "hierarchical_main_severity_model.keras"
FILE_ID = "1hyA0U4B2ePaaHlj2rIns9HR2BhmqR3Cg"   # your real file id

@st.cache_resource(show_spinner=True)
def load_smartseva_model():

    # If model file doesn't exist → download it once
    if not os.path.exists(MODEL_PATH):
        st.warning("⬇️ Downloading SmartSeva model from Google Drive… please wait…")

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        # fuzzy=True makes it bypass Google Drive virus scan warnings
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

        st.success("✅ Model downloaded successfully!")

    # Load model safely
    st.info("⏳ Loading AI model…")
    mdl = load_model(MODEL_PATH, compile=False)
    st.success("🚀 Model loaded successfully!")
    return mdl


# Load model only ONCE (cached — no re-download & no crashes)
model = load_smartseva_model()


# =====================================================================================
#                               BASIC APP SETUP
# =====================================================================================
SAVE_DIR = 'smartpraja_uploaded'
os.makedirs(SAVE_DIR, exist_ok=True)

main_classes = ['garbage', 'road', 'child']
severity_classes = [
    'low_garbage', 'heavy_garbage', 'low_damage_roads',
    'high_damage_roads', 'normal_child', 'child_labour'
]

ALERT_INFO = {
    "garbage": {"department": "Municipal Corporation & Sanitation Department"},
    "road": {"department": "PWD, NHAI, Public Works Department"},
    "child": {"department": "Child Welfare Committee / DCPU"}
}


# =====================================================================================
#                               IMAGE PREPROCESSING
# =====================================================================================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image or empty file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img


# =====================================================================================
#                                   PREDICTION
# =====================================================================================
def predict_issue(img_path):
    img_array, img = preprocess_image(img_path)
    main_pred, sev_pred = model.predict(img_array)

    main_label = main_classes[np.argmax(main_pred)]
    sev_label = severity_classes[np.argmax(sev_pred)]

    confidence = float(max(np.max(main_pred), np.max(sev_pred)) * 100)

    return sev_label, confidence, main_label, img


# =====================================================================================
#                               ALERT & FEEDBACK
# =====================================================================================
def generate_alert(pred_class, confidence, main_label):

    if "garbage" in pred_class:
        issue = "Garbage / Waste"
        alert_to = ALERT_INFO["garbage"]
        emergency = "🔴 HIGH" if "heavy" in pred_class else "🟡 MEDIUM"
        time = "few hours" if "heavy" in pred_class else "within a day"

    elif "roads" in pred_class:
        issue = "Road Damage"
        alert_to = ALERT_INFO["road"]
        emergency = "🔴 HIGH" if "high" in pred_class else "🟡 MEDIUM"
        time = "few hours" if "high" in pred_class else "within a day"

    elif "child" in pred_class:
        issue = "Child Case"
        if "labour" in pred_class:
            alert_to = ALERT_INFO["child"]
            emergency = "🔴 HIGH"
            time = "few hours"
        else:
            alert_to = {"department": "None"}
            emergency = "🟢 No Action"
            time = None

    else:
        issue = "Unknown"
        alert_to = {"department": "Unknown"}
        emergency = "⚪ Manual Check"
        time = None

    # Feedback
    if "child_labour" in pred_class:
        feedback = "⚠️ Possible child labour. Alert sent to Childline 1098."
    elif "normal_child" in pred_class:
        feedback = "✅ Normal child. No action required."
    else:
        feedback = f"Thanks for reporting. Officials will respond {time}."

    return {
        "Issue Type": issue,
        "Predicted Class": pred_class,
        "Confidence": f"{confidence:.2f}%",
        "Emergency Level": emergency,
        "Department": alert_to["department"],
        "Feedback": feedback
    }


# =====================================================================================
#                                     STREAMLIT UI
# =====================================================================================
st.title("📢 SmartSeva – AI Complaint Categorization")
st.write("Upload an image to classify garbage issues, road damage, or child labour cases.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    save_path = os.path.join(SAVE_DIR, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Processing…"):
            try:
                pred_class, confidence, main_label, img = predict_issue(save_path)
                result = generate_alert(pred_class, confidence, main_label)

                st.subheader("📊 Prediction Result")
                st.write(f"**Main Category:** {result['Issue Type']}")
                st.write(f"**Predicted Sub-Class:** {result['Predicted Class']}")
                st.write(f"**Confidence:** {result['Confidence']}")
                st.write(f"**Emergency Level:** {result['Emergency Level']}")
                st.write(f"**Department:** {result['Department']}")

                st.subheader("💬 Feedback")
                st.success(result["Feedback"])

            except Exception as e:
                st.error(f"❌ Error: {e}")
