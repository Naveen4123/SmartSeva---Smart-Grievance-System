# 🚦 SmartSeva – AI-Powered Grievance Detection

SmartSeva is an AI-driven grievance classification system designed to help users report civic issues quickly and accurately. By simply uploading an image, the system predicts **Category** (Road / Garbage / Child Issue) and **Severity Level** (Low / High), giving a fast, automated assessment right on the user’s screen.

---

## ✨ Features

- 🔍 **AI Image Classification** – Detects category of the issue from the uploaded image  
- ⚠️ **Severity Prediction** – Estimates issue severity using a second dedicated AI model  
- 📤 **Image Upload System** – Easy drag-and-drop or manual upload  
- 🧠 **Smart Validation** – User verifies prediction (YES / NO) to improve accuracy  
- ⚡ **Fast Inference** – ONNX-optimized models for speed  
- 💡 **Clean UI** – Simple, user-friendly interface

---

## 📂 Categories Predicted  
- 🛣️ **Road Issue**  
- 🗑️ **Garbage Issue**  
- 👶 **Child Issue**

---

## 🔥 Severity Levels  
- 🟩 **Low**  
- 🟥 **High**

---

## 🏗️ System Workflow

1. User uploads an image  
2. Model-1 predicts **Category**  
3. Model-2 predicts **Severity**  
4. Results displayed instantly  
5. User confirms correctness  
6. If incorrect → user re-uploads or tries again  

---

## 📢 User Confirmation Message

**"Here are the predicted category and severity for your uploaded image. If this looks correct, click YES. If not, click NO to re-upload or try again."**

---

## 🚀 Future Enhancements

- 🧩 **Add More Issue Categories**  
  Expand beyond current three classes with additional datasets.

- 📊 **Dashboard & Analytics**  
  Admin panel to view number of reports by location, type, severity.

- 🌐 **Location Auto-Detection**  
  Use GPS or EXIF metadata to capture issue location.

- 🗺️ **Heatmap of Issues**  
  Visualize problem areas in real time.

- 📝 **Text + Image Hybrid Feedback**  
  Allow users to describe the issue along with the image.

- 🤖 **Fine-Tuned Larger Models**  
  Upgrading from lightweight CNNs to MobileViT / EfficientNet.

- 🔄 **Active Learning Loop**  
  User feedback used to retrain the model for higher accuracy.

- 📱 **Mobile App Version**  
  Android/iOS app for on-the-go reporting.

- 🧾 **Auto-Generated Issue Summary**  
  AI creates a formatted grievance message for government portals.

---

## 🛠️ Tech Stack

- **Python** (Model Training)  
- **TensorFlow / PyTorch**  
- **ONNX Runtime** for fast inference  
- **Flask / FastAPI** (Backend)  
- **HTML / CSS / JS** (Frontend)  
- **GitHub** (Version Control)

---

## 🧪 Model Details

### 🎯 Category Classification Model  
- Input: Image (224×224)  
- Output: 3 Classes → Road / Garbage / Child  
- Architecture: Custom CNN / Lightweight Model  
- Format: ONNX  

### ⚠️ Severity Prediction Model  
- Input: Image (224×224)  
- Output: Low / High  
- Architecture: Custom CNN  
- Format: ONNX  

---

 ## 👥 Team Members

### 🧠 Data Science Team  
- **NaveenKumarReddy Bapathi**  
- **Anitha Sirigireddy**

### 💻 Full Stack Team  
- **Gautham**  
- **Likitha**  
- **Pavani**

