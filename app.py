import os
import csv
import subprocess
from pathlib import Path
import cv2
import pickle

from flask import (
    Flask,
    request,
    send_from_directory,
    url_for,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from recognize_number import recognize_number


# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / "boxes"
ANSWER_KEY_DIR = BASE_DIR / "answer_keys"
RESULTS_DIR = BASE_DIR / "results"

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}

INPUT_DIR.mkdir(exist_ok=True)
ANSWER_KEY_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)  # ✅ ENABLE CORS


# ---------------- LOAD DIGIT MODEL ----------------

MODEL_PATH = BASE_DIR / "mnist_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

with open(MODEL_PATH, "rb") as f:
    digit_model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    digit_scaler = pickle.load(f)


# ---------------- HELPERS ----------------

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def run_cropping_script():
    subprocess.run(["python", "extract_answer_boxes.py"], check=True)


def load_labels():
    labels_path = OUTPUT_DIR / "labels.csv"
    results = []

    if not labels_path.exists():
        return results

    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "question": int(row["question"]),
                "raw_path": row["path"]
            })

    return results


def load_answer_key(pdf_path):
    from extract_key import extract_answer_key
    return extract_answer_key(pdf_path)[1]


def evaluate_cropped_answer(image_path, correct_answer):
    image = cv2.imread(str(image_path))
    if image is None:
        return "Unable to read", 0.0, ""

    predicted, confidence = recognize_number(
        image,
        digit_model,
        digit_scaler
    )

    if "?" in predicted or predicted.strip() == "":
        remark = "Unable to read"
    elif str(predicted) == str(correct_answer):
        remark = "Correct"
    else:
        remark = "Wrong"

    return remark, round(confidence * 100, 2), predicted


# ---------------- ROUTES ----------------

@app.route("/process", methods=["POST"])
def process():

    # Clear old inputs
    for f in INPUT_DIR.glob("*"):
        try:
            f.unlink()
        except:
            pass

    image_file = request.files.get("answer_sheet")
    answer_key_file = request.files.get("answer_key")

    if not image_file or image_file.filename == "":
        return {"error": "No answer sheet uploaded"}, 400

    if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return {"error": "Invalid image file type"}, 400

    # Save answer sheet
    image_filename = secure_filename(image_file.filename)
    image_path = INPUT_DIR / image_filename
    image_file.save(image_path)

    # Load answer key
    answer_key_dict = {}
    if answer_key_file and answer_key_file.filename != "":
        if allowed_file(answer_key_file.filename, ALLOWED_PDF_EXTENSIONS):
            pdf_filename = secure_filename(answer_key_file.filename)
            pdf_path = ANSWER_KEY_DIR / pdf_filename
            answer_key_file.save(pdf_path)
            answer_key_dict = load_answer_key(pdf_path)

    # Run cropping
    run_cropping_script()

    # Load cropped cells
    cropped_data = load_labels()

    merged_results = []
    total_questions = 0
    total_correct = 0

    for item in cropped_data:
        q = item["question"]
        full_path = Path(item["raw_path"])
        relative_path = full_path.relative_to(OUTPUT_DIR).as_posix()

        correct_answer = answer_key_dict.get(q, "—")

        remark = "—"
        confidence = 0.0
        predicted = ""

        if correct_answer != "—":
            remark, confidence, predicted = evaluate_cropped_answer(
                full_path,
                correct_answer
            )

            total_questions += 1
            if remark == "Correct":
                total_correct += 1

        merged_results.append({
            "question": q,
            "image_url": url_for("serve_output", filename=relative_path),
            "correct_answer": correct_answer,
            "detected_answer": predicted,
            "remark": remark,
            "confidence": confidence
        })

    accuracy = 0.0
    if total_questions > 0:
        accuracy = round((total_correct / total_questions) * 100, 2)

    # Save CSV
    csv_path = RESULTS_DIR / "evaluation_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question",
            "Detected Answer",
            "Correct Answer",
            "Remark",
            "Confidence"
        ])

        for item in merged_results:
            writer.writerow([
                item["question"],
                item["detected_answer"],
                item["correct_answer"],
                item["remark"],
                item["confidence"]
            ])

    return {
        "results": merged_results,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "accuracy": accuracy
    }


@app.route("/download_results")
def download_results():
    return send_from_directory(
        RESULTS_DIR,
        "evaluation_results_corrected.csv",
        as_attachment=True
    )

@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/save_corrections", methods=["POST"])
def save_corrections():
    data = request.json
    corrected_results = data.get("results", [])

    total_questions = 0
    total_correct = 0

    csv_path = RESULTS_DIR / "evaluation_results_corrected.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question",
            "Detected Answer",
            "Correct Answer",
            "Remark",
            "Confidence"
        ])

        for item in corrected_results:

            confidence_value = (
                "Manually Corrected"
                if item.get("manuallyEdited")
                else item["confidence"]
            )

            writer.writerow([
                item["question"],
                item["detected_answer"],
                item["correct_answer"],
                item["remark"],
                confidence_value
            ])

            if item["correct_answer"] != "—":
                total_questions += 1
                if item["remark"] == "Correct":
                    total_correct += 1

    accuracy = 0.0
    if total_questions > 0:
        accuracy = round((total_correct / total_questions) * 100, 2)

    return {
        "message": "Corrections saved",
        "total_correct": total_correct,
        "total_questions": total_questions,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    app.run(debug=True)
