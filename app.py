from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from inference import predict_from_bytes

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per spec

app = Flask(__name__)


def _run_prediction():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Supported formats: JPG, JPEG, PNG only."
        }), 400

    image_bytes = file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "File exceeds 10MB limit"}), 400

    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file"}), 400

    try:
        result = predict_from_bytes(image_bytes)
        confidence_pct = round(result["confidence"] * 100, 1)
        label = "Pneumonia" if result["prediction"] == "PNEUMONIA" else "Normal"

        return jsonify({
            "prediction": label,
            "confidence": f"{confidence_pct}%",
            "confidence_value": confidence_pct,
            "pneumonia_probability": result["pneumonia_probability"],
            "raw_prediction": result["prediction"],
        })
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    return _run_prediction()


if __name__ == "__main__":
    print("[OK] PneumoScan: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
