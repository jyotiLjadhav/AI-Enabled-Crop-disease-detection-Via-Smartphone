# Android APK (WebView wrapper)

This folder contains a minimal Android WebView app that loads the Flask web app URL.

## Steps

1. Open `android_webview/` in Android Studio (it will download Gradle).
2. Update the URL in `android_webview/app/src/main/java/com/plantdetect/webview/MainActivity.kt:23`.
   - Emulator: keep `http://10.0.2.2:5000`
   - Real phone on same Wi‑Fi: `http://<your-laptop-LAN-IP>:5000`
   - Production: `https://your-domain/`
3. Run the app.

Notes:
- WebView can use mic/camera permissions for the Voice Assistant and image capture.
- For production, host the Flask app over HTTPS (recommended) and change the URL.

