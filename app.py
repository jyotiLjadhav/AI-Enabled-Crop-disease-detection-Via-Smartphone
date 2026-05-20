import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import sqlite3
import json
from datetime import datetime
import uuid
from typing import Optional, List, Dict
import cv2
import random
from fpdf import FPDF
import io
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect, send_file, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'predictions.sqlite3'
app.config['SECRET_KEY'] = 'agri-guard-secret-key-123' # Change in production
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- AUTH LOGIC ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email, is_admin=0):
        self.id = id
        self.username = username
        self.email = email
        self.is_admin = bool(is_admin)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Admin access required.", "danger")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    with _get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if user:
            return User(user['id'], user['username'], user['email'], user['is_admin'])
    return None

# --- DATABASE LOGIC ---
def _get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                created_at TEXT,
                image_filename TEXT,
                disease TEXT,
                confidence REAL,
                top3_json TEXT,
                gradcam_filename TEXT,
                weather_temp REAL,
                weather_humidity REAL,
                lat REAL,
                lon REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        # Handle cases where the table exists but lacks newer columns
        cursor = conn.execute("PRAGMA table_info(predictions)")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if 'lat' not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN lat REAL")
        if 'lon' not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN lon REAL")
        if 'user_id' not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
        
        # Handle is_admin column migration
        cursor = conn.execute("PRAGMA table_info(users)")
        user_cols = [row[1] for row in cursor.fetchall()]
        if 'is_admin' not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
            
        # Global Settings Table
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        # Initialize broadcast message if empty
        exists = conn.execute("SELECT * FROM settings WHERE key = 'broadcast_message'").fetchone()
        if not exists:
            conn.execute("INSERT INTO settings (key, value) VALUES ('broadcast_message', 'Welcome to AgriGuard AI. Stay safe and monitor your crops!')")
            
        # AUTO-CREATE MASTER ADMIN
        admin_user = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
        hashed_pw = generate_password_hash('admin123')
        if not admin_user:
            conn.execute("INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
                        ('admin', hashed_pw, 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print("Master Admin account 'admin' created successfully.")
        else:
            # Force update password to admin123 just in case
            conn.execute("UPDATE users SET password_hash = ?, is_admin = 1 WHERE username = 'admin'", (hashed_pw,))

def _record_prediction(
    image_filename: str,
    disease: str,
    confidence: float,
    top3: List[Dict],
    gradcam_filename: Optional[str],
    weather_temp: Optional[float],
    weather_humidity: Optional[float],
    lat: Optional[float],
    lon: Optional[float],
    user_id: Optional[int] = None,
):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top3_json = json.dumps(top3, ensure_ascii=False)
    with _get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (user_id, created_at, image_filename, disease, confidence, top3_json, gradcam_filename, weather_temp, weather_humidity, lat, lon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, created_at, image_filename, disease, confidence, top3_json, gradcam_filename, weather_temp, weather_humidity, lat, lon),
        )
        return cursor.lastrowid

_init_db()

# --- MODEL LOADING ---
MODEL_PATH = "model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- GRAD-CAM LOGIC ---
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    with open("gradcam_debug.log", "a") as f:
        f.write(f"{datetime.now()}: Starting make_gradcam_heatmap for {last_conv_layer_name}\n")
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        found = False
        for layer in model.layers:
            if found: x = layer(x)
            if layer.name == last_conv_layer_name: found = True
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            if pred_index is None: pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        if grads is None:
            with open("gradcam_debug.log", "a") as f: f.write("Gradients are None!\n")
            return np.zeros((224, 224))

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0: max_val = 1e-10
        heatmap = heatmap / max_val
        
        with open("gradcam_debug.log", "a") as f: f.write(f"Heatmap generated. Max val: {max_val.numpy()}\n")
        return heatmap.numpy()
    except Exception as e:
        with open("gradcam_debug.log", "a") as f: f.write(f"Error in make_gradcam_heatmap: {str(e)}\n")
        return np.zeros((225, 225))

def save_and_overlay_gradcam(img_path, model, last_conv_layer_name, pred_index, out_path):
    img = cv2.imread(img_path)
    if img is None: 
        with open("gradcam_debug.log", "a") as f: f.write(f"Image not found at {img_path}\n")
        return None
    img_array = get_img_array(img_path, size=(225, 225)) / 255.0
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    # Increase heatmap visibility (60% heatmap, 40% image)
    superimposed_img = jet * 0.6 + img * 0.4
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    success = cv2.imwrite(out_path, superimposed_img)
    with open("gradcam_debug.log", "a") as f: f.write(f"Image saved to {out_path}: {success}\n")
    return out_path

def is_leaf_image(img_path):
    """
    Heuristic to check if an image likely contains a plant leaf.
    Uses HSV color filtering to detect the density of 'organic' colors (green/yellow/brown).
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    # Convert to HSV color space for more robust color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a broad range for plant-like colors (Greens, Yellows, Browns)
    # Hue: 20-90 (Greenish), Saturation: >20, Value: >20
    lower_organic = np.array([20, 20, 20])
    upper_organic = np.array([95, 255, 255])

    # Create a mask
    mask = cv2.inRange(hsv, lower_organic, upper_organic)

    # Calculate the ratio of organic pixels to total pixels
    organic_pixel_count = np.sum(mask > 0)
    total_pixels = img.shape[0] * img.shape[1]
    organic_ratio = organic_pixel_count / total_pixels

    # Log for debugging
    with open("gradcam_debug.log", "a") as f:
        f.write(f"Validation: Organic pixel ratio for {os.path.basename(img_path)} is {organic_ratio:.4f}\n")

    # If less than 12% of the image has leaf-like colors, it's probably not a leaf
    return organic_ratio > 0.12

# --- AUTH ROUTES ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        with _get_db() as conn:
            existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                flash("Username already exists!", "danger")
                return redirect(url_for('register'))
            
            pwd_hash = generate_password_hash(password)
            conn.execute("INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                         (username, email, pwd_hash, datetime.now().isoformat()))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with _get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if user and check_password_hash(user['password_hash'], password):
                user_obj = User(user['id'], user['username'], user['email'], user['is_admin'])
                login_user(user_obj)
                flash(f"Welcome back, {username}!", "success")
                return redirect(url_for('index'))
            else:
                flash("Invalid username or password", "danger")
                
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

# --- MAIN ROUTES ---
@app.route('/')
def index():
    with _get_db() as conn:
        broadcast = conn.execute("SELECT value FROM settings WHERE key = 'broadcast_message'").fetchone()
        broadcast_msg = broadcast['value'] if broadcast else "Welcome to AgriGuard AI!"
    return render_template('index.html', broadcast_msg=broadcast_msg, show_hero=True)

@app.route('/detect')
@login_required
def detect():
    with _get_db() as conn:
        broadcast = conn.execute("SELECT value FROM settings WHERE key = 'broadcast_message'").fetchone()
        broadcast_msg = broadcast['value'] if broadcast else "Welcome to AgriGuard AI!"
    return render_template('index.html', broadcast_msg=broadcast_msg, show_hero=False)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if model is None: return jsonify({"error": "Model not loaded"}), 500

    # --- PRE-VALIDATION ---
    if not is_leaf_image(file_path):
        # We don't record invalid attempts in history, but we delete the temp file
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            "error": "Input is not a leaf.",
            "suggestion": "The AI could not detect plant characteristics in this image. Please upload a clear photo of a plant leaf."
        }), 400

    # Classification
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(225, 225))
    x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    predictions = model.predict(x)[0]
    labels = ["Healthy", "Powdery", "Rust"] 
    top_indices = predictions.argsort()[-3:][::-1]
    
    lang = request.form.get('lang', 'en')
    
    descriptions = {
        'en': {
            'Healthy': "The plant exhibits excellent physiological health with vibrant green pigmentation and robust leaf structure. There are no detectable traces of fungal pathogens, necrotic lesions, or chlorotic patterns. The vascular system appears fully functional, ensuring optimal nutrient distribution. We recommend maintaining current irrigation and soil nutrition protocols to sustain this high level of vitality and resilience against seasonal stressors.",
            'Powdery': "Powdery Mildew (Erysiphales) is a specialized fungal infection manifesting as white, flour-like mycelial growth on leaf surfaces. It typically starts as small circular spots and rapidly expands to cover the entire canopy, significantly reducing the plant's photosynthetic capacity. If left untreated, it leads to leaf curling, premature yellowing, and eventual tissue necrosis. This disease thrives in moderate temperatures and high humidity, often affecting yield quality and total biomass.",
            'Rust': "Leaf Rust is a complex fungal disease caused by pathogens of the order Pucciniales, characterized by small, elevated orange-to-brown pustules. These pustules contain millions of microscopic spores that are easily dispersed by wind and water, leading to rapid field-wide outbreaks. The infection punctures the leaf epidermis, causing significant moisture loss and disrupting the plant's metabolic balance. Severe cases result in mass defoliation, stunted growth, and a drastic reduction in crop productivity."
        },
        'kn': {
            'Healthy': "ಗಿಡವು ಅತ್ಯಂತ ಆರೋಗ್ಯಕರವಾಗಿದೆ ಮತ್ತು ಎಲೆಗಳು ಗಾಢ ಹಸಿರು ಬಣ್ಣದಿಂದ ಕೂಡಿದೆ. ಯಾವುದೇ ರೀತಿಯ ಶಿಲೀಂಧ್ರ ಅಥವಾ ರೋಗದ ಲಕ್ಷಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ. ಗಿಡದ ಪೋಷಕಾಂಶಗಳ ವಿತರಣಾ ವ್ಯವಸ್ಥೆಯು ಉತ್ತಮವಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿದೆ. ಇದೇ ರೀತಿಯ ನೀರಾವರಿ ಮತ್ತು ಗೊಬ್ಬರದ ನಿರ್ವಹಣೆಯನ್ನು ಮುಂದುವರಿಸಿ. ಇದು ಗಿಡದ ರೋಗನಿರೋಧಕ ಶಕ್ತಿಯನ್ನು ಹೆಚ್ಚಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.",
            'Powdery': "ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ಎಂಬುದು ಒಂದು ಶಿಲೀಂಧ್ರ ರೋಗವಾಗಿದ್ದು, ಇದು ಎಲೆಗಳ ಮೇಲೆ ಬಿಳಿ ಬಣ್ಣದ ಹಿಟ್ಟಿನಂತಹ ಚುಕ್ಕೆಗಳ ರೂಪದಲ್ಲಿ ಕಾಣಿಸಿಕೊಳ್ಳುತ್ತದೆ. ಇದು ಕ್ರಮೇಣ ಇಡೀ ಎಲೆಯನ್ನು ಆವರಿಸಿ ಗಿಡದ ದ್ಯುತಿಸಂಶ್ಲೇಷಣಾ ಕ್ರಿಯೆಯನ್ನು ಕುಂಠಿತಗೊಳಿಸುತ್ತದೆ. ಸರಿಯಾದ ಸಮಯದಲ್ಲಿ ಚಿಕಿತ್ಸೆ ನೀಡದಿದ್ದರೆ ಎಲೆಗಳು ಹಳದಿಯಾಗಿ ಉದುರಿಹೋಗುತ್ತವೆ. ಇದು ಸಾಮಾನ್ಯವಾಗಿ ಹೆಚ್ಚಿನ ಆರ್ದ್ರತೆ ಮತ್ತು ಸಾಧಾರಣ ತಾಪಮಾನದಲ್ಲಿ ವೇಗವಾಗಿ ಹರಡುತ್ತದೆ ಮತ್ತು ಇಳುವರಿಯನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತದೆ.",
            'Rust': "ರಸ್ಟ್ ಅಥವಾ ತುಕ್ಕು ರೋಗವು ಪುಸ್ಸಿನಿಯಲ್ಸ್ ಎಂಬ ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ, ಇದು ಎಲೆಗಳ ಮೇಲೆ ಕಿತ್ತಳೆ ಅಥವಾ ಕಂದು ಬಣ್ಣದ ಸಣ್ಣ ಗುಳ್ಳೆಗಳಂತೆ ಕಾಣಿಸಿಕೊಳ್ಳುತ್ತದೆ. ಈ ಗುಳ್ಳೆಗಳಲ್ಲಿ ಲಕ್ಷಾಂತರ ಸೂಕ್ಷ್ಮಾಣುಗಳಿದ್ದು, ಇವು ಗಾಳಿ ಮತ್ತು ನೀರಿನ ಮೂಲಕ ಇಡೀ ಹೊಲಕ್ಕೆ ಹರಡುತ್ತವೆ. ಈ ರೋಗವು ಎಲೆಯ ಮೇಲ್ಮೈಯನ್ನು ಹಾನಿಗೊಳಿಸಿ ಗಿಡದ ನೀರಿನ ಅಂಶವನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತದೆ. ಇದರಿಂದ ಗಿಡದ ಬೆಳವಣಿಗೆ ಕುಂಠಿತವಾಗಿ ಇಳುವರಿಯಲ್ಲಿ ಭಾರಿ ಕುಸಿತ ಉಂಟಾಗುತ್ತದೆ."
        },
        'hi': {
            'Healthy': "पौधा पूरी तरह से स्वस्थ है और इसमें कोई भी रोग के लक्षण नहीं हैं। पत्तियों का रंग गहरा हरा है और संरचना मजबूत है। पौधे की पोषण प्रणाली सुचारू रूप से कार्य कर रही है। पौधों की वर्तमान सिंचाई और खाद प्रबंधन को जारी रखें। यह पौधों को मौसमी तनाव के खिलाफ लचीला बनाए रखने में मदद करेगा।",
            'Powdery': "पाउडरी मिल्ड्यू एक कवक संक्रमण है जो पत्तियों की सतह पर सफेद, आटे जैसे धब्बों के रूप में दिखाई देता है। यह धीरे-धीरे पूरी पत्ती को ढक लेता है और पौधे की प्रकाश संश्लेषण क्षमता को कम कर देता है। यदि अनुपचारित छोड़ दिया जाए, तो पत्तियां पीली होकर गिरने लगती हैं। यह बीमारी मध्यम तापमान और उच्च आर्द्रता में तेजी से फैलती है, जिससे फसल की गुणवत्ता और पैदावार प्रभावित होती है।",
            'Rust': "रस्ट या रतुआ रोग एक कवक रोग है जो पत्तियों पर छोटे, उभरे हुए नारंगी या भूरे रंग के धब्बों के रूप में दिखाई देता है। इन धब्बों में लाखों बीजाणु होते हैं जो हवा और पानी के माध्यम से पूरे खेत में फैल सकते हैं। यह संक्रमण पत्ती की सतह को नुकसान पहुंचाता है और पौधे की नमी को कम कर देता है। गंभीर मामलों में, इसके कारण पत्तियां समय से पहले गिर जाती हैं और विकास रुक जाता है।"
        }
    }
    remedies = {
        'en': {
            'Healthy': "1. Continue regular moisture monitoring to prevent water stress.\n2. Apply balanced organic fertilizers to maintain soil nutrient levels.\n3. Prune dense foliage to ensure optimal sunlight penetration.\n4. Keep the surroundings clean to prevent potential pest habitats.\n5. Conduct routine scouting every 48 hours for early detection.",
            'Powdery': "1. Apply Sulfur-based fungicides or Neem oil immediately upon detection.\n2. Improve air circulation by thinning out dense plant canopies.\n3. Avoid overhead irrigation to reduce leaf surface moisture.\n4. Remove and safely dispose of heavily infected leaves to stop spore spread.\n5. Use potassium bicarbonate sprays for environment-friendly control.",
            'Rust': "1. Spray systemic fungicides like Mancozeb or Copper Oxychloride.\n2. Remove all infected crop debris and weeds that harbor spores.\n3. Avoid working in the fields when leaves are wet to prevent spreading.\n4. Transition to rust-resistant crop varieties in the next planting cycle.\n5. Ensure balanced nitrogen application to avoid excessive soft growth."
        },
        'kn': {
            'Healthy': "1. ಗಿಡಕ್ಕೆ ಸರಿಯಾದ ಸಮಯದಲ್ಲಿ ನೀರು ಹಾಯಿಸುವುದನ್ನು ಮುಂದುವರಿಸಿ.\n2. ಮಣ್ಣಿನ ಫಲವತ್ತತೆ ಕಾಪಾಡಲು ಸಮತೋಲಿತ ಸಾವಯವ ಗೊಬ್ಬರ ನೀಡಿ.\n3. ಗಿಡಕ್ಕೆ ಸರಿಯಾಗಿ ಗಾಳಿ ಮತ್ತು ಬೆಳಕು ಸಿಗುವಂತೆ ನೋಡಿಕೊಳ್ಳಿ.\n4. ಕಳೆಗಳನ್ನು ಕಿತ್ತು ಹಾಕಿ ಗಿಡದ ಸುತ್ತಮುತ್ತ ಸ್ವಚ್ಛವಾಗಿಡಿ.\n5. ಪ್ರತಿ 48 ಗಂಟೆಗೊಮ್ಮೆ ಗಿಡದ ಆರೋಗ್ಯವನ್ನು ಗಮನಿಸುತ್ತಿರಿ.",
            'Powdery': "1. ರೋಗ ಕಂಡ ತಕ್ಷಣ ಗಂಧಕ ಆಧಾರಿತ ಅಥವಾ ಬೇವು ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ.\n2. ಗಿಡಗಳ ನಡುವೆ ಹೆಚ್ಚಿನ ಅಂತರವಿರುವಂತೆ ನೋಡಿಕೊಳ್ಳಿ.\n3. ಎಲೆಗಳ ಮೇಲೆ ನೇರವಾಗಿ ನೀರು ಸುರಿಯುವುದನ್ನು ತಪ್ಪಿಸಿ.\n4. ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ಕಿತ್ತು ಹೊಲದಿಂದ ದೂರ ಹಾಕಿ ಅಥವಾ ಸುಟ್ಟು ಹಾಕಿ.\n5. ಗಾಳಿ ಸಂಚಾರ ಸುಧಾರಿಸಲು ದಟ್ಟವಾಗಿ ಬೆಳೆದ ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ.",
            'Rust': "1. ಮ್ಯಾಂಕೋಜೆಬ್ ಅಥವಾ ತಾಮ್ರದ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ.\n2. ರೋಗಪೀಡಿತ ಎಲೆಗಳು ಮತ್ತು ತ್ಯಾಜ್ಯವನ್ನು ಹೊಲದಿಂದ ಸಂಪೂರ್ಣವಾಗಿ ತೆಗೆಯಿರಿ.\n3. ಎಲೆಗಳು ಒದ್ದೆಯಾಗಿರುವಾಗ ಹೊಲದಲ್ಲಿ ಕೆಲಸ ಮಾಡಬೇಡಿ.\n4. ಮುಂದಿನ ಬಾರಿ ರೋಗನಿರೋಧಕ ತಳಿಗಳನ್ನು ಆಯ್ಕೆ ಮಾಡಿ.\n5. ಅತಿಯಾದ ಸಾರಜನಕ ಬಳಕೆಯನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಪೊಟ್ಯಾಶ್ ಗೊಬ್ಬರ ಬಳಸಿ."
        },
        'hi': {
            'Healthy': "1. पौधों को पानी का तनाव रोकने के लिए नियमित सिंचाई जारी रखें।\n2. मिट्टी के पोषक तत्वों को बनाए रखने के लिए जैविक खाद का उपयोग करें।\n3. पौधों की छंटाई करें ताकि उन्हें पर्याप्त धूप और हवा मिल सके।\n4. पौधों के आसपास की जगह को साफ रखें ताकि कीट न पनप सकें।\n5. हर 2-3 दिन में पौधों की जांच करते रहें ताकि किसी भी बदलाव का पता चले।",
            'Powdery': "1. रोग के लक्षण दिखते ही सल्फर आधारित कवकनाशी या नीम के तेल का छिड़काव करें।\n2. पौधों के बीच हवा के संचार को बेहतर बनाने के लिए सघन टहनियों की छंटाई करें।\n3. पत्तियों के ऊपर सीधे पानी डालने से बचें।\n4. गंभीर रूप से संक्रमित पत्तियों को हटा दें और नष्ट कर दें ताकि बीजाणु न फैलें।\n5. पर्यावरण अनुकूल नियंत्रण के लिए पोटेशियम बाइकार्बोनेट का उपयोग करें।",
            'Rust': "1. मैनकोजेब या कॉपर ऑक्सीक्लोराइड जैसे कवकनाशी का छिड़काव करें।\n2. संक्रमित फसल के अवशेषों और खरपतवारों को खेत से पूरी तरह हटा दें।\n3. जब पत्तियां गीली हों तो खेत में काम करने से बचें ताकि बीमारी न फैले।\n4. अगली बुवाई के लिए रोग प्रतिरोधी किस्मों का चयन करें।\n5. नाइट्रोजन का संतुलित उपयोग करें ताकि अत्यधिक कोमल वृद्धि न हो।"
        }
    }

    top_results = []
    for idx in top_indices:
        label = labels[idx]
        prob = float(predictions[idx]) * 100.0
        top_results.append({
            "label": label,
            "confidence": prob,
            "description": descriptions.get(lang, descriptions['en']).get(label, ""),
            "remedy": remedies.get(lang, remedies['en']).get(label, "")
        })

    predicted_label = labels[top_indices[0]]
    confidence = float(predictions[top_indices[0]]) * 100.0
    description = descriptions.get(lang, descriptions['en']).get(predicted_label, "")
    remedy = remedies.get(lang, remedies['en']).get(predicted_label, "")

    # --- DYNAMIC WEATHER ADVICE ---
    weather_data_raw = request.form.get('weather_data')
    humidity = 60 # Default
    temp = 25 # Default
    
    try:
        if weather_data_raw:
            import json
            w = json.loads(weather_data_raw)
            humidity = w.get('main', {}).get('humidity', 60)
            temp = w.get('main', {}).get('temp', 25)
    except: pass

    if predicted_label == 'Healthy':
        weather_advice = f"Current weather (Temp: {temp}°C, Humidity: {humidity}%) is optimal for plant growth. Continue regular monitoring."
    elif predicted_label == 'Powdery':
        if humidity > 70:
            weather_advice = f"CRITICAL: High humidity ({humidity}%) detected. Powdery Mildew spreads rapidly in these conditions. Apply Sulfur spray immediately."
        else:
            weather_advice = "Dry conditions detected. Powdery Mildew thrives in low moisture; ensure adequate spacing for air circulation."
    elif predicted_label == 'Rust':
        if temp > 20:
            weather_advice = f"WARNING: Warm temperatures ({temp}°C) will accelerate Rust spore germination. Apply protective fungicides before the next rain."
        else:
            weather_advice = "Cooler temperatures detected, but keep leaves dry as Rust requires only 2-4 hours of surface moisture to infect."
    else:
        weather_advice = "Normal weather conditions. Monitor local forecasts for sudden changes in humidity."

    last_conv_layer_name = next((l.name for l in reversed(model.layers) if 'conv' in l.name), None)
    gradcam_url = None
    if last_conv_layer_name:
        try:
            gradcam_filename = f"gradcam_{uuid.uuid4().hex}.jpg"
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            with open("gradcam_debug.log", "a") as f: f.write(f"Processing Grad-CAM for {filename} -> {gradcam_filename}\n")
            save_and_overlay_gradcam(file_path, model, last_conv_layer_name, int(top_indices[0]), gradcam_path)
            gradcam_url = url_for('uploaded_file', filename=gradcam_filename)
        except Exception as e:
            with open("gradcam_debug.log", "a") as f: f.write(f"Predict Route Error: {str(e)}\n")
            print(f"Grad-CAM Error: {e}")

    # Use the dynamic weather_advice generated above

    pred_id = _record_prediction(
        image_filename=filename,
        disease=predicted_label,
        confidence=confidence,
        top3=top_results,
        gradcam_filename=os.path.basename(gradcam_url) if gradcam_url else None,
        weather_temp=temp,
        weather_humidity=humidity,
        lat=request.form.get('lat', type=float),
        lon=request.form.get('lon', type=float),
        user_id=current_user.id
    )

    classifiers = [
        {"name": "Custom CNN", "confidence": confidence, "status": "Primary"},
        {"name": "MobileNetV2", "confidence": max(0, min(100, confidence + random.uniform(-5, 5))), "status": "Simulated"},
        {"name": "InceptionV3", "confidence": max(0, min(100, confidence + random.uniform(-8, 2))), "status": "Simulated"}
    ]

    return jsonify({
        "id": pred_id, "result": predicted_label, "confidence": confidence, "description": description,
        "remedy": remedy, "top3": top_results, "gradcam": gradcam_url, "weather_advice": weather_advice, "classifiers": classifiers
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
@login_required
def history():
    with _get_db() as conn:
        items = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC", (current_user.id,)).fetchall()
        items = [dict(r) for r in items]
        
        # Add image URLs for template
        for item in items:
            item['image_url'] = url_for('uploaded_file', filename=item['image_filename'])

        total = len(items)
        
        # Most common disease
        counts = {}
        for it in items:
            d = it['disease']
            counts[d] = counts.get(d, 0) + 1
        most_common = max(counts, key=counts.get) if counts else None

        # Chart Data (Trend)
        # Group by date (YYYY-MM-DD)
        trend_data = {}
        for it in items:
            date_str = it['created_at'].split(' ')[0]
            trend_data[date_str] = trend_data.get(date_str, 0) + 1
        
        sorted_dates = sorted(trend_data.keys())
        chart = {
            "labels": sorted_dates,
            "datasets": [{
                "label": "Daily Predictions",
                "data": [trend_data[d] for d in sorted_dates],
                "borderColor": "#4caf50",
                "backgroundColor": "rgba(76, 175, 80, 0.1)",
                "fill": True,
                "tension": 0.4
            }]
        }

    return render_template('history.html', items=items, total=total, most_common=most_common, chart=chart)

@app.route('/history/clear', methods=['POST'])
@login_required
def history_clear():
    with _get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE user_id = ?", (current_user.id,))
    return redirect(url_for('history'))

@app.route('/history/delete/<int:pred_id>', methods=['POST'])
@login_required
def history_delete(pred_id):
    with _get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?", (pred_id, current_user.id))
    return redirect(url_for('history'))

@app.route('/download_report/<int:pred_id>')
@login_required
def download_report(pred_id):
    try:
        with _get_db() as conn:
            row = conn.execute("SELECT * FROM predictions WHERE id = ? AND user_id = ?", (pred_id, current_user.id)).fetchone()
        
        if not row:
            return "Report not found", 404

        row = dict(row)

        # Initialize PDF in A4
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        
        # Header Styling
        pdf.set_fill_color(27, 94, 32) # Dark Green
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", 'B', 24)
        pdf.cell(0, 20, "AgriGuard AI Report", ln=True, align='C')
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 5, "Official Crop Health Diagnostic Certificate", ln=True, align='C')
        pdf.ln(15)

        # Body Text Styling
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", 'I', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='R')
        pdf.ln(5)
        
        def add_safe_row(label, value):
            pdf.set_font("Helvetica", 'B', 11)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(50, 10, f" {label}", border=1, fill=True)
            pdf.set_font("Helvetica", '', 11)
            # Filter non-latin characters for standard Helvetica
            safe_val = str(value).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 10, f" {safe_val}", border=1, ln=True)

        add_safe_row("Prediction ID", f"#{row['id']}")
        add_safe_row("Date/Time", row['created_at'])
        add_safe_row("Detected Disease", row['disease'])
        add_safe_row("Confidence Score", f"{row.get('confidence', 0):.2f}%")
        
        if row.get('lat') and row.get('lon'):
            add_safe_row("Location", f"{row['lat']}, {row['lon']}")
        
        if row.get('weather_temp'):
            add_safe_row("Weather Info", f"{row['weather_temp']}C | {row['weather_humidity']}% Humid")

        pdf.ln(10)
        
        # Image Handling
        img_filename = row.get('image_filename')
        if img_filename:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            if os.path.exists(img_path):
                pdf.set_font("Helvetica", 'B', 14)
                pdf.set_text_color(27, 94, 32)
                pdf.cell(0, 10, "Analyzed Specimen Image", ln=True)
                pdf.ln(2)
                pdf.image(img_path, x=15, w=90)
                pdf.ln(5)

        # Detailed Analysis
        if row.get('top3_json'):
            # Safe vertical offset after image
            current_y = pdf.get_y()
            if current_y < 120: pdf.set_y(150) # Ensure we don't overlap the image
            
            pdf.set_font("Helvetica", 'B', 14)
            pdf.set_text_color(27, 94, 32)
            pdf.cell(0, 10, "Detailed AI Diagnosis", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            try:
                top3 = json.loads(row['top3_json'])
                for res in top3:
                    pdf.set_font("Helvetica", 'B', 11)
                    pdf.cell(0, 8, f"> {res['label']}: {res['confidence']:.2f}%", ln=True)
                    pdf.set_font("Helvetica", '', 10)
                    
                    desc = res.get('description', 'N/A').encode('latin-1', 'replace').decode('latin-1')
                    rem = res.get('remedy', 'N/A').encode('latin-1', 'replace').decode('latin-1')
                    
                    pdf.multi_cell(0, 5, f"Info: {desc}")
                    pdf.multi_cell(0, 5, f"Remedy: {rem}")
                    pdf.ln(3)
            except:
                pdf.cell(0, 10, "Detailed candidate data unavailable.", ln=True)

        # Footer
        pdf.set_y(-25)
        pdf.set_font("Helvetica", 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 4, "Disclaimer: This report is generated by AgriGuard AI. Always verify with local agricultural experts before applying treatments.", align='C')

        pdf_output = pdf.output()
        return send_file(
            io.BytesIO(pdf_output),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"AgriGuard_Report_{row['id']}.pdf"
        )
    except Exception as e:
        print(f"PDF GENERATION ERROR: {e}")
        return f"Could not generate PDF: {str(e)}", 500

@app.route('/history_data')
@login_required
def history_data():
    with _get_db() as conn:
        rows = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC", (current_user.id,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/map')
@login_required
def map_view(): return render_template('map.html')

@app.route('/map_data')
@login_required
def map_data():
    real_data = []
    with _get_db() as conn:
        rows = conn.execute("SELECT lat, lon, disease, created_at FROM predictions WHERE lat IS NOT NULL").fetchall()
        real_data = [dict(r) for r in rows]
    
    # Generate 40 Realistic Demo Points for the presentation
    import random
    from datetime import datetime, timedelta
    demo_points = []
    base_lat, base_lon = 15.3647, 75.1240 # Hubli Center
    diseases = ['Leaf Rust', 'Powdery Mildew', 'Healthy Plant']
    
    for _ in range(40):
        d_lat = random.uniform(-0.04, 0.04)
        d_lon = random.uniform(-0.04, 0.04)
        disease = random.choice(diseases)
        # Random time in last 48 hours
        time_offset = random.randint(0, 2880) 
        time_str = (datetime.now() - timedelta(minutes=time_offset)).strftime("%Y-%m-%d %H:%M:%S")
        
        demo_points.append({
            'lat': base_lat + d_lat,
            'lon': base_lon + d_lon,
            'disease': disease,
            'created_at': time_str,
            'is_demo': True
        })
    
    return jsonify(real_data + demo_points)

# --- ADMIN ROUTES ---
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    with _get_db() as conn:
        # Total Stats
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_scans = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        
        # Analytics: Disease Distribution (Pie Chart)
        disease_counts = conn.execute("SELECT disease, COUNT(*) as count FROM predictions GROUP BY disease ORDER BY count DESC").fetchall()
        disease_labels = [row['disease'] for row in disease_counts]
        disease_values = [row['count'] for row in disease_counts]
        
        # Analytics: Usage Trends (Daily Scans)
        trend_data = conn.execute("""
            SELECT date(created_at) as scan_date, COUNT(*) as count 
            FROM predictions 
            GROUP BY scan_date 
            ORDER BY scan_date ASC 
            LIMIT 30
        """).fetchall()
        trend_labels = [row['scan_date'] for row in trend_data]
        trend_values = [row['count'] for row in trend_data]

        # All Users
        users_list = conn.execute("SELECT id, username, email, is_admin, created_at FROM users ORDER BY created_at DESC").fetchall()
        
        # All Scans (with username)
        scans_list = conn.execute("""
            SELECT p.*, u.username 
            FROM predictions p 
            LEFT JOIN users u ON p.user_id = u.id 
            ORDER BY p.created_at DESC
        """).fetchall()

        # Current Broadcast
        broadcast = conn.execute("SELECT value FROM settings WHERE key = 'broadcast_message'").fetchone()
        broadcast_msg = broadcast['value'] if broadcast else ""
        
    return render_template('admin.html', 
                         total_users=total_users, 
                         total_scans=total_scans, 
                         users=users_list, 
                         scans=scans_list,
                         disease_labels=disease_labels,
                         disease_values=disease_values,
                         trend_labels=trend_labels,
                         trend_values=trend_values,
                         broadcast_msg=broadcast_msg)

@app.route('/admin/update_broadcast', methods=['POST'])
@admin_required
def admin_update_broadcast():
    msg = request.form.get('broadcast_msg', '')
    with _get_db() as conn:
        conn.execute("UPDATE settings SET value = ? WHERE key = 'broadcast_message'", (msg,))
        conn.commit()
    flash("Global Advisory updated successfully!", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/export_csv')
@admin_required
def admin_export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow(['ID', 'Farmer', 'Date', 'Disease', 'Confidence', 'Temp', 'Humidity', 'Lat', 'Lon'])
    
    with _get_db() as conn:
        scans = conn.execute("""
            SELECT p.id, u.username, p.created_at, p.disease, p.confidence, p.weather_temp, p.weather_humidity, p.lat, p.lon 
            FROM predictions p 
            LEFT JOIN users u ON p.user_id = u.id
        """).fetchall()
        
        for s in scans:
            writer.writerow([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]])
            
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'agri_guard_full_data_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').lower()
    
    # --- Intelligent Multi-Path Response Logic ---
    if 'rust' in user_msg:
        reply = "Leaf Rust is identified by orange or brown pustules on the leaf underside. I suggest removing infected leaves immediately and applying a copper-based fungicide. Avoid overhead watering as it spreads the spores!"
    elif 'powdery' in user_msg or 'white' in user_msg:
        reply = "Powdery Mildew looks like a dusting of flour on your plants. It thrives in high humidity. Try a mix of 1 part milk to 9 parts water as a natural spray, or use Neem oil for severe cases."
    elif 'spots' in user_msg or 'disease' in user_msg:
        reply = "Leaf spots can be bacterial or fungal. Since our AI detected patterns, please check the 'Remedy' section of your report for a precise treatment plan based on your specific crop."
    elif any(k in user_msg for k in ['weather', 'rain', 'humidity']):
        reply = "Weather is the biggest factor in disease spread. High humidity (>80%) significantly increases the risk of Rust. Use our Outbreak Map to see if your local area is currently at high risk."
    elif any(k in user_msg for k in ['organic', 'natural', 'home']):
        reply = "For organic control, use a spray of baking soda (1 tsp) and liquid soap (1/2 tsp) in 1 liter of water. Ensure your plants have proper spacing for better air circulation to prevent future outbreaks."
    elif any(k in user_msg for k in ['price', 'buy', 'product', 'market']):
        reply = "Check the 'Recommended Products' section in your scan results! We've automatically found the best marketplace links for the fungicides and fertilizers needed for your specific case."
    elif any(k in user_msg for k in ['hi', 'hello', 'hey', 'namaste']):
        reply = "Hello! I am your AgriGuard AI assistant. I can explain Rust symptoms, suggest organic sprays, or help you with weather-based risks. What can I do for you?"
    elif any(k in user_msg for k in ['thank', 'thanks', 'helpful', 'welcome']):
        reply = "You're very welcome! I'm happy to help you protect your crops. Remember, early detection is the key to a healthy harvest. Feel free to ask me anything else!"
    else:
        reply = "That's a great question! As an AI expert, I focus on early detection. If you have a photo of the affected plant, please upload it for a deep analysis. I can also help with organic remedies or market prices."
        
    return jsonify({"reply": reply})

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    app.run(host='0.0.0.0', port=port, debug=True)
