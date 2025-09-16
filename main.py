import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Read threshold (default to 0.5 if not set)
THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.56))
FAKE_THRESHOLD = float(os.getenv("FAKE_DETECTION_THRESHOLD", 0.45))
# Load your Keras model
model = load_model("deepfake_efficient.keras")

# def preprocess_image(file, target_size=(224, 224)):

#     # Read file bytes
#     file_bytes = np.frombuffer(file.read(), np.uint8)

#     # Decode image with OpenCV (force 3 channels BGR)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # --- Step 1: Convert to LAB and apply CLAHE on L-channel ---
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)

#     limg = cv2.merge((cl, a, b))
#     img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#     # --- Step 2: Sharpening ---
#     sharpen_kernel = np.array([
#         [0, -1, 0],
#         [-1, 5, -1],
#         [0, -1, 0]
#     ])
#     img = cv2.filter2D(img, -1, sharpen_kernel)

#     # --- Step 3: Resize ---
#     img = cv2.resize(img, target_size)

#     # --- Step 4: Convert BGR → RGB ---
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # --- Step 5: Normalize ---
#     img = img.astype("float32") / 255.0

#     # --- Step 6: Add batch dimension ---
#     img = np.expand_dims(img, axis=0)  # shape (1,224,224,3)

#     return img

def preprocess_image_original(file_bytes, target_size=(224, 224)):
    debug_images = {}
    # Read file bytes
    # file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    debug_images["original"] = img.copy()
    
    img = cv2.resize(img, target_size)
    
    debug_images["resized"] = img.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    debug_images["Color_converted"] = (img * 255).astype("uint8")
    img = img.astype("float32") / 255.0
    debug_images["final"] = (img * 255).astype("uint8")
    processed = np.expand_dims(img, axis=0)
    return processed , debug_images

def preprocess_image(file_bytes, target_size=(224, 224)):
    
    # Read file bytes
    # file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR

    debug_images = {}

    # Original
    debug_images["original"] = img.copy()

    # --- Step 1: CLAHE ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    debug_images["clahe"] = img_clahe.copy()

    # --- Step 2: Sharpening ---
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img_sharpened = cv2.filter2D(img_clahe, -1, sharpen_kernel)
    debug_images["sharpened"] = img_sharpened.copy()

    # --- Step 3: Resize ---
    img_resized = cv2.resize(img_sharpened, target_size)
    debug_images["resized"] = img_resized.copy()

    # --- Step 4: Convert BGR → RGB & Normalize ---
    img_final = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    debug_images["Color_converted"] = (img_final * 255).astype("uint8")
    img_final = img_final.astype("float32") / 255.0
    
    debug_images["final"] = (img_final * 255).astype("uint8")
    processed = np.expand_dims(img_final, axis=0)  # for model
    
    return processed, debug_images

@app.route("/")
def home():
    return "Deepfake Detection API is running."

@app.route("/debug_preprocess", methods=["POST"])
def debug_preprocess():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    processed, debug_images = preprocess_image(file)

    # Save debug images so you can open them in browser
    saved_paths = {}
    for name, img in debug_images.items():
        path = f"static/{name}.jpg"
        cv2.imwrite(path, img)  # save as JPG
        saved_paths[name] = path

    return jsonify({
        "message": "Images saved for debugging",
        "paths": saved_paths
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    print("file received")
    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    # processed , _ = preprocess_image(file)
    processed, debug_images = preprocess_image(file_bytes)
    print("image preprocessed")

    # Save debug images so you can open them in browser
    saved_paths = {}
    for name, img in debug_images.items():
        path = f"static/{name}.jpg"
        cv2.imwrite(path, img)  # save as JPG
        # saved_paths[name] = path
        saved_paths[name] = path
        
    prediction = model.predict(processed)
    
    print("prediction made")
    print(prediction)
    
    if prediction <= THRESHOLD and prediction >= FAKE_THRESHOLD:
        # file.seek(0)
        processed ,debug_images= preprocess_image_original(file_bytes)
        # Save debug images so you can open them in browser
        saved_paths = {}
        for name, img in debug_images.items():
            path = f"origin_static/{name}.jpg"
            cv2.imwrite(path, img)  # save as JPG
            # saved_paths[name] = path
            saved_paths[name] = path
        print("Reprocessing with original method")
        prediction = model.predict(processed)
        print("New prediction:", prediction)
        
        return jsonify({"prediction": prediction.tolist(),
                    "message": "Images saved for debugging with original preprocessing",
                    "paths": saved_paths})
        
        
    return jsonify({"prediction": prediction.tolist(),
                    "message": "Images saved for debugging",
                    "paths": saved_paths})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
