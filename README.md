# 📢 SmartSeva AI – Smart Grievance Categorization System

SmartSeva AI is a deep-learning powered image classification system designed to automatically detect and categorize public complaints such as **garbage issues, road damage, and child-related cases**.  
Using a **hierarchical multi-task deep learning model**, SmartSeva predicts both:

1️⃣ **Main Category**  
2️⃣ **Severity Sub-Class**

This helps authorities respond to complaints faster, more accurately, and with proper priority levels.

---

# 🚀 Features

- 🧠 **AI-powered classification (EfficientNet Backbone)**
- 🔥 **Two-head hierarchical multi-task model**
- 🔽 **Auto-downloads `.keras` model from Google Drive**
- ⚡ **Runs smoothly on Streamlit Cloud**
- 👁️ **Real-time image analysis**
- 📊 **Shows classes, confidence, severity, emergency level**
- 🏛️ **Maps the complaint to the correct government department**
- 🟢 **Lightweight model (<100MB) for fast deployment**

---

# 🧠 Model Architecture

SmartSeva uses a **Hierarchical Multi-Task Learning (H-MTL)** structure.

### 🔹 **Base Model**
- **EfficientNetB0**
- Image size: **224×224×3**
- Weights: **ImageNet Pretrained**

### 🔹 **Output Heads**
#### 1. **Main Category (3 classes)**
- garbage  
- road  
- child  

#### 2. **Severity Classification (6 classes)**
- low_garbage  
- heavy_garbage  
- low_damage_roads  
- high_damage_roads  
- normal_child  
- child_labour  

### 🔹 **Training Details**
- Optimizer: **Adam (1e-4)**
- Loss (multi-output):
- Augmentations: flip, rotate, zoom, brightness
- 10–20 epochs with EarlyStopping

---

# 🏷 Class Breakdown

### **Main Classes (3)**

| Class   | Meaning |
|--------|---------|
| garbage | Waste, dump yard, trash piles |
| road | Road cracks, potholes, damage |
| child | Normal child or child labour case |

---

### **Severity Sub-Classes (6)**

| Sub-Class | Category | Meaning |
|-----------|----------|---------|
| low_garbage | Garbage | Small garbage |
| heavy_garbage | Garbage | Heavy waste dumping |
| low_damage_roads | Roads | Minor cracks |
| high_damage_roads | Roads | Severe potholes / major damage |
| normal_child | Child | Regular child image |
| child_labour | Child | Possible child labour (alert) |

---

# 🧰 Technologies Used

- Python  
- TensorFlow 2.20  
- Keras  
- EfficientNet  
- OpenCV  
- Streamlit  
- NumPy  
- Pillow  
- gdown (Google Drive model downloader)

---

# 📂 Dataset (General Overview)

SmartSeva uses ~2,100 images collected from:

- Google Images  
- Manual collection  
- Open datasets  
- Field images  

All images were resized to **224x224**, normalized, and augmented.

---

# 📁 Project Structure
SmartSeva/
│
├── application.py # Streamlit app
├── requirements.txt # Dependencies
├── README.md # Documentation
├── smartpraja_uploaded/ # Auto-created folder for uploaded files
└── hierarchical_main_severity_model.keras (downloaded automatically)

---

# 🧪 How the App Works

1️⃣ User uploads an image  
2️⃣ Image is resized and preprocessed using EfficientNet preprocess  
3️⃣ Model performs two predictions:
   - Main Category  
   - Severity Class  
4️⃣ SmartSeva maps outputs to:
   - Emergency level (High/Medium/None)
   - Government department
   - Action feedback  
5️⃣ Streamlit displays results in a clean UI

---

# 💻 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/smartseva.git
cd smartseva
