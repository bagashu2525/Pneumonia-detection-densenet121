from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from inference import predict_from_bytes

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 25 * 1024 * 1024

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Upload a .jpg, .jpeg, or .png chest X-ray image."
        }), 400

    image_bytes = file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "File exceeds 25MB limit"}), 400

    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file"}), 400

    try:
        result = predict_from_bytes(image_bytes)
        return jsonify({
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "pneumonia_probability": result["pneumonia_probability"],
        })
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


if __name__ == "__main__":
    print("[OK] Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
