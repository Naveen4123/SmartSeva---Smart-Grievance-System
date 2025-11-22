import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# ===================== LOAD MODEL =====================
MODEL_PATH = "hierarchical_main_severity_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# ===================== CLASS LABELS =====================
main_classes = ['garbage', 'road', 'child']

severity_classes = [
    'low_garbage',
    'heavy_garbage',
    'low_damage_roads',
    'high_damage_roads',
    'normal_child',
    'child_labour'
]

# ===================== PREPROCESS IMAGE =====================
def preprocess_image(image):
    image = image.resize((224, 224))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

# ===================== PREDICT FUNCTION =====================
def predict_image(image):
    """
    Input: PIL image
    Output: main_category, sub_class, confidence
    """

    img_array = preprocess_image(image)

    # Model returns two outputs → main + severity
    main_pred, sev_pred = model.predict(img_array)

    # Getting indexes
    main_idx = np.argmax(main_pred)
    sev_idx = np.argmax(sev_pred)

    main_label = main_classes[main_idx]
    severity_label = severity_classes[sev_idx]

    # Confidence from the highest score among both heads
    confidence = max(np.max(main_pred), np.max(sev_pred)) * 100

    return main_label, severity_label, confidence
