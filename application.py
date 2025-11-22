import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ==================== PATHS ====================
MODEL_PATH = 'hierarchical_main_severity_model.keras'
SAVE_DIR = 'smartpraja_uploaded'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

# Classes
main_classes = ['garbage', 'road', 'child']
severity_classes = ['low_garbage', 'heavy_garbage', 'low_damage_roads',
                    'high_damage_roads', 'normal_child', 'child_labour']

# Alert info
ALERT_INFO = {
    "garbage": {"department": "Municipal Corporation & Sanitation Department"},
    "road": {"department": "PWD, NHAI, Public Works Department"},
    "child": {"department": "Child Welfare Committee / DCPU"}
}


# ==================== IMAGE PREPROCESS ====================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img


# ==================== PREDICTION ====================
def predict_issue(img_path):
    img_array, img = preprocess_image(img_path)
    main_pred, sev_pred = model.predict(img_array)

    main_idx = np.argmax(main_pred)
    sev_idx = np.argmax(sev_pred)

    main_label = main_classes[main_idx]
    sev_label = severity_classes[sev_idx]

    confidence = max(np.max(main_pred), np.max(sev_pred)) * 100

    return sev_label, confidence, main_label, img


# ==================== ALERT & FEEDBACK ====================
def generate_alert(pred_class, confidence, main_label):

    # GARBAGE
    if "garbage" in pred_class:
        issue_type = "Garbage / Waste"
        alert = ALERT_INFO["garbage"]
        emergency = "🔴 HIGH EMERGENCY" if "heavy" in pred_class else "🟡 MEDIUM PRIORITY"
        timeline = "within a few hours" if "heavy" in pred_class else "within a day"

    # ROADS
    elif "roads" in pred_class:
        issue_type = "Road Damage"
        alert = ALERT_INFO["road"]
        emergency = "🔴 HIGH EMERGENCY" if "high" in pred_class else "🟡 MEDIUM PRIORITY"
        timeline = "within a few hours" if "high" in pred_class else "within a day"

    # CHILD ISSUES
    elif "child" in pred_class:
        issue_type = "Child-related Case"

        if "labour" in pred_class:
            alert = ALERT_INFO["child"]
            emergency = "🔴 HIGH EMERGENCY"
            timeline = "within a few hours"
        else:
            alert = {"department": "None"}
            emergency = "🟢 NO ACTION REQUIRED"
            timeline = None

    else:
        issue_type = "Unknown"
        alert = {"department": "Unknown"}
        emergency = "⚪ Check Manually"
        timeline = None

    # Feedback message
    if "child_labour" in pred_class:
        feedback = (
            "⚠️ The uploaded image indicates possible child labour.\n"
            "An urgent alert has been sent to the Labour Department / Childline 1098."
        )

    elif "normal_child" in pred_class:
        feedback = (
            "✅ The child in the photo is not a Child Labour.\n"
            "No action required. Thank you for using SmartSeva!"
        )

    elif "high" in pred_class or "heavy" in pred_class:
        feedback = (
            f"✅ Thanks for using SmartSeva!\n"
            f"They will contact you via Phone or Mail {timeline}."
        )

    else:
        feedback = (
            f"✅ Thanks for using SmartSeva!\n"
            f"They will contact you via Phone or Mail {timeline}."
        )

    return {
        "Issue Type": issue_type,
        "Predicted Class": pred_class,
        "Confidence": f"{confidence:.2f}%",
        "Emergency Level": emergency,
        "Department": alert["department"],
        "Feedback": feedback
    }


# ==================== STREAMLIT APP ====================
st.title("📢 SmartSeva – AI Complaint Categorization System")
st.write("Upload an image to detect garbage, road damage, or child-related issues.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save file
    save_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Processing..."):
            try:
                pred_class, confidence, main_label, img = predict_issue(save_path)
                result = generate_alert(pred_class, confidence, main_label)

                st.subheader("📊 Prediction Result")
                st.write(f"**Main Category:** {result['Issue Type']}")
                st.write(f"**Sub Class:** {result['Predicted Class']}")
                st.write(f"**Confidence:** {result['Confidence']}")
                st.write(f"**Emergency Level:** {result['Emergency Level']}")
                st.write(f"**Department:** {result['Department']}")

                st.subheader("💬 Feedback")
                st.success(result["Feedback"])

            except Exception as e:
                st.error(f"Error: {e}")
