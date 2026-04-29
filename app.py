# Grad-CAM utility (optional: heatmap overlay requires matplotlib)
import json
import os
import sys
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from flask import redirect
from PIL import Image as PILImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# --- FIX: Define app and paths before using them ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
os.makedirs(USER_DATA_DIR, exist_ok=True)

try:
    import matplotlib.cm as cm
except Exception:
    cm = None


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def _leaf_exg_ratio(img: PILImage.Image, *, size: int = 128) -> float:
    """
    Heuristic "looks like a leaf/vegetation" score.

    The disease model is trained on leaf photos and will still output a class for
    non-leaf images (overconfident softmax). This check reduces obvious misuse
    (e.g., room/wall selfies) by requiring some vegetation-like pixels.
    """
    im = img.convert("RGB").resize((size, size))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    exg = 2.0 * g - r - b  # Excess Green index
    veg_mask = (exg > 0.08) & (g > 0.20)
    return float(np.mean(veg_mask))

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input as the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    # Iterate through layers after the last conv layer
    found = False
    for layer in model.layers:
        if found:
            x = layer(x)
        if layer.name == last_conv_layer_name:
            found = True
    classifier_model = tf.keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        
        # Compute predictions
        preds = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    if grads is None:
        return None

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    return heatmap

def save_and_overlay_gradcam(image_path, model, last_conv_layer_name, pred_index=None, out_path=None):
    """
    Generate and save Grad-CAM heatmap overlay on the original image.
    """
    if not cm:
        print("Grad-CAM: Matplotlib CM not available.")
        return None
    
    try:
        # Load and prepare image for model
        img = PILImage.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Standardize resize to model's expected input
        img_resized = img.resize((225, 225))
        img_array = img_to_array(img_resized)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
        
        if heatmap is None:
            print("Grad-CAM: Heatmap generation returned None.")
            return None
            
        heatmap_np = heatmap
        
        # Check if heatmap is all zeros (failed to track gradients properly)
        if np.max(heatmap_np) <= 0:
            print("Grad-CAM: Generated heatmap is empty/zero.")
            return None

        # Colorize heatmap
        heatmap_img = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=1), cmap='jet')
        heatmap_colored = heatmap_img.to_rgba(heatmap_np)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Resize heatmap back to original image size
        heatmap_pil = PILImage.fromarray(heatmap_colored)
        heatmap_pil = heatmap_pil.resize(img.size, PILImage.LANCZOS)
        
        # Overlay heatmap on original image (50/50 blend)
        img_np = np.array(img)
        heatmap_np_array = np.array(heatmap_pil)
        overlay = (0.5 * img_np + 0.5 * heatmap_np_array).astype(np.uint8)
        result_img = PILImage.fromarray(overlay)
        
        # Save result
        if out_path:
            result_img.save(out_path, 'JPEG', quality=95)
            # Verify file exists
            if os.path.exists(out_path):
                return out_path
        
        return None
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return None
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
app.config["LEAF_EXG_RATIO_MIN"] = float(os.environ.get("LEAF_EXG_RATIO_MIN", "0.02"))

# History DB should be writable (works in source + bundled EXE)
app.config["HISTORY_DB"] = os.path.join(USER_DATA_DIR, "predictions.sqlite3")

# Load trained model (from bundled resources)
model = load_model(os.path.join(BASE_DIR, "model.h5"))

print("Model loaded successfully.")

# Labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Remedies for each disease

# Remedies in multiple languages
remedies = {
    'en': {
        'Healthy': 'No action needed. Keep monitoring your crop.',
        'Powdery': 'Apply fungicide and remove affected leaves.',
        'Rust': 'Use appropriate fungicide and avoid overhead watering.'
    },
    'hi': {
        'Healthy': 'कोई कार्रवाई आवश्यक नहीं है। अपनी फसल की निगरानी करते रहें।',
        'Powdery': 'फफूंदी नाशक का छिड़काव करें और प्रभावित पत्तियों को हटा दें।',
        'Rust': 'उपयुक्त फफूंदी नाशक का उपयोग करें और ऊपर से पानी देने से बचें।'
    },
    'kn': {
        'Healthy': 'ಯಾವುದೇ ಕ್ರಮ ಅಗತ್ಯವಿಲ್ಲ. ನಿಮ್ಮ ಬೆಳೆಗಳನ್ನು ಗಮನಿಸಿ.',
        'Powdery': 'ಫಂಗಿಸೈಡ್ ಅನ್ನು ಹಚ್ಚಿ ಮತ್ತು ಪ್ರಭಾವಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.',
        'Rust': 'ಸರಿಯಾದ ಫಂಗಿಸೈಡ್ ಬಳಸಿ ಮತ್ತು ಮೇಲಿನಿಂದ ನೀರು ಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ.'
    }
}

# Descriptions for each disease in multiple languages
descriptions = {
    'en': {
        'Healthy': 'The leaf appears to be healthy with no visible signs of disease. The plant is growing normally.',
        'Powdery': 'Powdery mildew is a fungal disease that affects a wide range of plants, appearing as white powdery spots on the leaves and stems.',
        'Rust': 'Rust is a fungal disease that causes rust-colored, orange, or yellow spots, usually on the undersides of leaves, which can stunt plant growth.'
    },
    'hi': {
        'Healthy': 'पत्ता स्वस्थ दिखता है, कोई रोग के लक्षण नहीं हैं। पौधा सामान्य रूप से बढ़ रहा है।',
        'Powdery': 'पाउडरी मिल्ड्यू एक फफूंदी रोग है जो कई पौधों को प्रभावित करता है, पत्तियों और तनों पर सफेद पाउडर जैसे धब्बे बनते हैं।',
        'Rust': 'रस्ट एक फफूंदी रोग है जिसमें पत्तियों के नीचे जंग जैसे, नारंगी या पीले धब्बे बनते हैं, जिससे पौधे की वृद्धि रुक सकती है।'
    },
    'kn': {
        'Healthy': 'ಎಲೆ ಆರೋಗ್ಯವಾಗಿದ್ದು ಯಾವುದೇ ರೋಗದ ಲಕ್ಷಣಗಳು ಕಾಣಿಸುತ್ತಿಲ್ಲ. ಸಸ್ಯವು ಸಾಮಾನ್ಯವಾಗಿ ಬೆಳೆಯುತ್ತಿದೆ.',
        'Powdery': 'ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ಎಂಬುದು ಹಲವಾರು ಸಸ್ಯಗಳನ್ನು ಪರಿಣಾಮಗೊಳಿಸುವ ಶಿಲೀಂಧ್ರ ರೋಗವಾಗಿದೆ, ಇದು ಎಲೆಗಳು ಮತ್ತು ಕಾಂಡಗಳಲ್ಲಿ ಬಿಳಿ ಪುಡಿ ಹಚ್ಚಿದಂತೆ ಕಾಣಿಸುತ್ತದೆ.',
        'Rust': 'ರಸ್ಟ್ ಎಂಬುದು ಎಲೆಗಳ ಕೆಳಭಾಗದಲ್ಲಿ ಕಿತ್ತಳೆ ಅಥವಾ ಹಳದಿ ಬಣ್ಣದ ಕಲೆಗಳನ್ನು ಉಂಟುಮಾಡುವ ಶಿಲೀಂಧ್ರ ರೋಗವಾಗಿದೆ, ಇದು ಸಸ್ಯದ ಬೆಳವಣಿಗೆಯನ್ನು ತಡೆಯಬಹುದು.'
    }
}

# Create uploads folder if not exists (writable)
UPLOAD_FOLDER = os.path.join(USER_DATA_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def _get_db():
    conn = sqlite3.connect(app.config["HISTORY_DB"])
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL,
                top3_json TEXT,
                gradcam_filename TEXT,
                weather_temp REAL,
                weather_humidity REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_disease ON predictions(disease)"
        )


def _record_prediction(
    *,
    image_filename: str,
    disease: str,
    confidence: Optional[float],
    top3: Optional[List[Dict[str, Any]]],
    gradcam_filename: Optional[str],
    weather_temp: Optional[float],
    weather_humidity: Optional[float],
):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top3_json = json.dumps(top3, ensure_ascii=False) if top3 is not None else None
    with _get_db() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (created_at, image_filename, disease, confidence, top3_json, gradcam_filename, weather_temp, weather_humidity)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                image_filename,
                disease,
                confidence,
                top3_json,
                gradcam_filename,
                weather_temp,
                weather_humidity,
            ),
        )


# Prediction function
def getResult(image_path):
    # Open image using PIL to ensure proper RGB conversion
    # which fixes issues with RGBA transparent PNGs affecting predictions.
    img = PILImage.open(image_path)
    if img.mode != 'RGB':
        # If the image has an alpha channel (like PNG), blend it with a white background
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = PILImage.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.convert('RGBA').split()[3])
            img = bg
        else:
            img = img.convert('RGB')
    
    img = img.resize((225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)[0]
    return predictions


# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/uploads/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/history", methods=["GET"])
def history():
    _init_db()

    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, image_filename, gradcam_filename, disease, confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()

        trend_rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, disease, COUNT(*) AS cnt
            FROM predictions
            GROUP BY day, disease
            ORDER BY day ASC
            """
        ).fetchall()

        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        most_common = conn.execute(
            """
            SELECT disease, COUNT(*) AS c
            FROM predictions
            GROUP BY disease
            ORDER BY c DESC
            LIMIT 1
            """
        ).fetchone()

    items = []
    for r in rows:
        items.append(
            {
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "disease": r["disease"],
                "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
                "image_url": url_for("uploaded_file", filename=r["image_filename"]),
            }
        )

    # Build chart.js friendly series: labels = days, datasets per disease.
    disease_order = ["Healthy", "Powdery", "Rust"]
    by_day = {}
    for tr in trend_rows:
        day = tr["day"]
        disease = tr["disease"]
        cnt = int(tr["cnt"])
        by_day.setdefault(day, {})[disease] = cnt

    days = sorted(by_day.keys())
    datasets = []
    palette = {
        "Healthy": "#2e7d32",
        "Powdery": "#f9a825",
        "Rust": "#c62828",
    }
    for disease in disease_order:
        datasets.append(
            {
                "label": disease,
                "data": [by_day.get(d, {}).get(disease, 0) for d in days],
                "borderColor": palette.get(disease, "#1976d2"),
                "backgroundColor": palette.get(disease, "#1976d2") + "22",
                "tension": 0.25,
                "fill": True,
            }
        )

    chart_payload = {"labels": days, "datasets": datasets}

    return render_template(
        "history.html",
        items=items,
        total=total,
        most_common=(most_common["disease"] if most_common else None),
        chart=chart_payload,
    )


def _safe_remove_upload(filename: Optional[str]) -> None:
    if not filename:
        return
    base = os.path.basename(filename)
    if not base:
        return
    path = os.path.join(app.config["UPLOAD_FOLDER"], base)
    try:
        os.remove(path)
    except Exception:
        pass


@app.post("/history/delete/<int:pred_id>")
def history_delete(pred_id: int):
    _init_db()
    with _get_db() as conn:
        row = conn.execute(
            "SELECT image_filename, gradcam_filename FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()
        if row is not None:
            conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))

    if row is not None:
        _safe_remove_upload(row["image_filename"])
        _safe_remove_upload(row["gradcam_filename"])

    return redirect(url_for("history"))


@app.post("/history/clear")
def history_clear():
    _init_db()
    with _get_db() as conn:
        files = conn.execute(
            "SELECT image_filename, gradcam_filename FROM predictions"
        ).fetchall()
        conn.execute("DELETE FROM predictions")

    for r in files:
        _safe_remove_upload(r["image_filename"])
        _safe_remove_upload(r["gradcam_filename"])

    return redirect(url_for("history"))



# Prediction route
@app.route('/predict', methods=['POST'])
def upload():
    _init_db()
    
    # Get language from request (default to 'en')
    lang = request.form.get('lang', 'en')
    if lang not in remedies:
        lang = 'en'
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    original_name = secure_filename(file.filename)
    if not _allowed_file(original_name):
        return jsonify({"error": "Unsupported file type. Upload JPG, PNG, or WEBP."}), 400

    _, ext = os.path.splitext(original_name.lower())
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Validate the uploaded file is an image.
    try:
        with PILImage.open(file_path) as im:
            im.verify()
    except Exception:
        try:
            os.remove(file_path)
        except Exception:
            pass
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    # Basic "leaf photo" sanity check to avoid obviously wrong inputs.
    try:
        with PILImage.open(file_path) as im:
            exg_ratio = _leaf_exg_ratio(im)
    except Exception:
        exg_ratio = 0.0

    if exg_ratio < app.config["LEAF_EXG_RATIO_MIN"]:
        try:
            os.remove(file_path)
        except Exception:
            pass
        return (
            jsonify(
                {
                    "error": (
                        "This doesn't look like a plant leaf photo. "
                        "Please upload/capture a clear leaf image (the model only knows: Healthy, Powdery, Rust)."
                    )
                }
            ),
            400,
        )


    predictions = getResult(file_path)
    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]
    top_results = []
    for idx in top_indices:
        idx_i = int(idx)
        label = labels[idx_i]
        prob = float(predictions[idx_i]) * 100.0
        top_results.append(
            {
                "label": label,
                "confidence": prob,
                "description": descriptions.get(lang, {}).get(label, "No description available."),
                "remedy": remedies.get(lang, {}).get(label, "No remedy available."),
            }
        )

    # Main result (top 1)
    predicted_label = labels[int(top_indices[0])]
    confidence = float(predictions[int(top_indices[0])]) * 100.0
    remedy = remedies.get(lang, {}).get(predicted_label, "No remedy available.")
    description = descriptions.get(lang, {}).get(predicted_label, "No description available.")

    # Grad-CAM (optional)
    last_conv_layer_name = next(
        (layer.name for layer in reversed(model.layers) if 'conv' in layer.name),
        None,
    )

    gradcam_url = None
    if cm is not None and last_conv_layer_name:
        try:
            gradcam_filename = f"gradcam_{uuid.uuid4().hex}.jpg"
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            saved_path = save_and_overlay_gradcam(
                file_path,
                model,
                last_conv_layer_name,
                pred_index=int(top_indices[0]),
                out_path=gradcam_path,
            )
            if saved_path:
                gradcam_url = url_for('uploaded_file', filename=gradcam_filename)
            else:
                gradcam_url = None
        except Exception:
            gradcam_url = None

    # Weather-based suggestion
    weather_humidity = request.form.get('weather_humidity', type=float)
    weather_temp = request.form.get("weather_temp", type=float)
    weather_advice = None
    if weather_humidity is not None:
        if weather_humidity > 80:
            weather_advice = "High humidity -> Powdery mildew risk increased. Monitor for fungal diseases."
        elif weather_humidity > 60:
            weather_advice = "Moderate humidity -> Fungal disease risk is moderate. Monitor your plants."
        else:
            weather_advice = "Low humidity -> Fungal disease risk is low."

    # Persist to history (best-effort; prediction must still succeed even if history fails)
    try:
        _record_prediction(
            image_filename=filename,
            disease=predicted_label,
            confidence=confidence,
            top3=top_results,
            gradcam_filename=(os.path.basename(gradcam_url) if gradcam_url else None),
            weather_temp=weather_temp,
            weather_humidity=weather_humidity,
        )
    except Exception:
        pass

    return jsonify(
        {
            "result": predicted_label,
            "confidence": confidence,
            "description": description,
            "remedy": remedy,
            "top3": top_results,
            "gradcam": gradcam_url,
            "weather_advice": weather_advice,
        }
    )


# Run app (IMPORTANT for mobile access)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    app.run(host='0.0.0.0', port=port, debug=True)
