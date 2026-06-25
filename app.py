from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pytesseract

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# 🔥 IMPORTANT: Change this path based on your installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================
# HOME PAGE
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# =========================
# UPLOAD + PROCESS IMAGE
# =========================
@app.route("/upload", methods=["POST"])
def upload():

    # 1. GET FILE FROM USER
    file = request.files["image"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    # =========================
    # 2. READ IMAGE (OPEN CV)
    # =========================
    image = cv2.imread(filepath)

    # =========================
    # 3. IMAGE PREPROCESSING
    # =========================

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # detect edges (document detection step)
    edges = cv2.Canny(blur, 50, 150)

    # =========================
    # 4. SAVE PROCESSED IMAGE
    # =========================
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], "processed_" + filename)
    cv2.imwrite(processed_path, edges)

    # =========================
    # 5. OCR (TEXT EXTRACTION)
    # =========================
    text = pytesseract.image_to_string(gray)

    # =========================
    # 6. SEND RESULT TO HTML
    # =========================
    return render_template(
        "result.html",
        image_path=processed_path,
        text=text
    )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)