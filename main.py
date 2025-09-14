from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load your Keras model
model = load_model("deepfake_efficient.keras")

def preprocess_image(img):
    # Resize to your model's input size (example: 224x224)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    return img_array
@app.route("/")
def home():
    return "Deepfake Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    print("file received")
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))

    processed = preprocess_image(img)
    print("image preprocessed")
    prediction = model.predict(processed)
    print("prediction made")
    print(prediction)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
