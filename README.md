# 🌿 AI-Enabled Crop Disease Detection via Smartphone

An intelligent deep learning–based web application that detects plant leaf diseases using smartphone images using AI and Computer Vision technologies.

---

# 📌 Problem Statement

Crop diseases significantly affect agricultural productivity and food quality.  
Farmers often struggle to identify plant diseases at an early stage due to:

- Lack of expert agricultural guidance
- Manual disease inspection methods
- Delay in disease diagnosis
- High crop loss caused by late treatment
- Difficulty accessing plant pathology experts in rural areas

Traditional disease detection methods are time-consuming, expensive, and less accessible for small-scale farmers.

Therefore, there is a need for an intelligent, fast, low-cost, and smartphone-based crop disease detection system.

---

# 💡 Proposed Solution

This project proposes an **AI-enabled smartphone-based crop disease detection system** using Deep Learning and Image Processing.

The system allows users to:

- Upload or capture plant leaf images using a smartphone
- Detect diseases automatically using a trained CNN model
- View prediction confidence scores
- Visualize infected regions using Grad-CAM heatmaps
- Receive weather-based farming suggestions
- Access the system through a simple web interface or Android APK

The solution helps farmers identify diseases quickly and take preventive actions at an early stage.

---

# 🎯 Objectives

- To develop a smart crop disease detection system using Deep Learning
- To classify plant leaf diseases accurately using CNN models
- To provide a user-friendly smartphone-based interface
- To reduce crop loss through early disease detection
- To integrate Explainable AI using Grad-CAM visualization
- To provide weather-aware agricultural recommendations
- To support offline UI functionality using PWA technology
- To create deployable Windows EXE and Android APK versions

---

## 📌 Features

✅ Upload plant leaf images from gallery  
✅ Capture images directly using smartphone camera  
✅ AI-powered disease prediction using CNN  
✅ Weather-based farming suggestions  
✅ Grad-CAM heatmap visualization (Explainable AI)  
✅ Progressive Web App (PWA) support  
✅ Offline UI support  
✅ Windows EXE generation using PyInstaller  
✅ Android APK support using WebView wrapper  

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| Flask | Web Framework |
| TensorFlow / Keras | Deep Learning Model |
| HTML / CSS / JavaScript | Frontend UI |
| OpenCV | Image Processing |
| Grad-CAM | Explainable AI Visualization |
| PyInstaller | Windows EXE Packaging |
| Android Studio | APK Generation |

---

# 📂 Project Structure

```bash
Plant-Disease-Detection/
│
├── app.py
├── model.h5
├── requirements.txt
├── requirements-gradcam.txt
├── test_predict.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── manifest.json
│   ├── service-worker.js
│   └── offline.html
│
├── Dataset/
│
├── packaging/
│   └── build_exe.ps1
│
└── android_webview/
```

---

# ⚙️ Installation Guide

## Clone Repository

```bash
git clone https://github.com/jyotiLjadhav/AI-Enabled-Crop-disease-detection-Via-Smartphone.git
```

## Navigate to Project Folder

```bash
cd AI-Enabled-Crop-disease-detection-Via-Smartphone
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

```bash
venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000/
```

---

# 📱 PWA Support

- Offline UI functionality
- Installable mobile-like experience
- Faster loading using service workers

---

# 🖥️ Build Windows EXE

```bash
powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1
```

---

# 📲 Android APK Build

1. Open `android_webview/` in Android Studio  
2. Update Flask server URL  
3. Build APK  

---

# 🧠 AI Model Information

- Model Type: CNN (Convolutional Neural Network)
- Framework: TensorFlow / Keras
- Input: Plant Leaf Image
- Output: Disease Prediction

---

# 🚀 Future Enhancements

- Real-time live camera detection
- Multi-language support
- Cloud deployment
- Fertilizer recommendation system
- Voice assistant integration

---

# 👩‍💻 Developer

**Jyoti Jadhav**  

GitHub: https://github.com/jyotiLjadhav

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub!
