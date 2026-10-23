import io
import os
import logging
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# ----- Logging: app.log file mein likhega -----
logging.basicConfig(
    filename=os.path.join(BASE, "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("oral-app")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device).eval()
labels = model.config.id2label   # {0: 'CANCER', 1: 'NON_CANCER'}
log.info("Model loaded on %s, classes=%s", device, list(labels.values()))

LOW_CONF_THRESHOLD = 70.0  # isse kam confidence par low_confidence flag

# Training ke test_transform jaisa preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = Flask(__name__, template_folder=os.path.join(BASE, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device), "classes": list(labels.values())})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        log.warning("Predict request without file")
        return jsonify({"error": "No image received."}), 400
    try:
        img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
    except Exception:
        log.warning("Invalid image file received")
        return jsonify({"error": "Invalid image file. Use a JPG or PNG photo."}), 400

    try:
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(pixel_values=x).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    except Exception:
        log.exception("Prediction failed")
        return jsonify({"error": "Prediction failed on the server."}), 500

    best = probs.index(max(probs))
    confidence = round(probs[best] * 100, 2)
    log.info("Prediction=%s confidence=%.2f low_conf=%s", labels[best], confidence, confidence < LOW_CONF_THRESHOLD)

    return jsonify({
        "prediction": labels[best],
        "confidence": confidence,
        "low_confidence": confidence < LOW_CONF_THRESHOLD,
        "probabilities": {labels[i]: round(p * 100, 2) for i, p in enumerate(probs)},
    })


@app.errorhandler(413)
def too_large(e):
    log.warning("Image rejected: larger than 10 MB")
    return jsonify({"error": "Image is too large. Use a photo under 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
