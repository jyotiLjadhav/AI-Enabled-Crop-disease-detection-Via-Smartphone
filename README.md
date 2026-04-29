# Plant Disease Detection (Flask + TensorFlow)

This project predicts plant leaf disease from an uploaded image using a trained CNN model (`model.h5`).
It also supports optional weather-based advice and optional Grad-CAM heatmap visualization.

## Project Structure

- `app.py`: Flask backend (model loading + `/predict` API)
- `templates/index.html`: Frontend UI (upload, camera capture, weather)
- `static/`: CSS/JS + PWA files
- `model.h5`: Trained TensorFlow/Keras model
- `Dataset/`: Sample train/val/test images (optional for testing)

## Setup (Windows)

1. Create a virtual environment (recommended):

   `python -m venv venv`

2. Activate:

   `venv\\Scripts\\Activate.ps1`

3. Install dependencies:

   `python -m pip install -r requirements.txt`

   Notes:
   - Grad-CAM overlay images require an extra install: `python -m pip install -r requirements-gradcam.txt`
   - TensorFlow on native Windows typically runs on CPU.

## Run

Start the server:

`python app.py`

Open in browser:

`http://127.0.0.1:5000/`

## Quick Test

You can test the model from the command line:

`python test_predict.py`

## PWA (Offline Support)

- The app includes `static/manifest.json` + `static/service-worker.js` + `static/offline.html`.
- Install it from Chrome/Edge using “Install app”.
- Offline mode supports opening pages and UI; prediction still requires a running backend.

## Build Windows EXE (.exe) with PyInstaller

1. Run: `powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1`
2. Start: `dist/PlantDetect/PlantDetect.exe`
3. Open: `http://127.0.0.1:5000/`

## Android APK (WebView Wrapper)

- Open `android_webview/` in Android Studio.
- Update the URL in `android_webview/app/src/main/java/com/plantdetect/webview/MainActivity.kt`.
- Build / Run to generate the APK.
