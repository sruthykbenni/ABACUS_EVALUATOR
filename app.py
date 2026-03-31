import csv
import os
import subprocess
from pathlib import Path

import cv2
import torch
from flask import Flask, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from recognize_number import recognize_number, recognize_numbers


# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / "boxes"
ANSWER_KEY_DIR = BASE_DIR / "answer_keys"
RESULTS_DIR = BASE_DIR / "results"

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}
MISSING_ANSWER = "-"
OCR_CONFIDENCE_THRESHOLD = 0.6

INPUT_DIR.mkdir(exist_ok=True)
ANSWER_KEY_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(
    app,
    origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
)


# ---------------- LOAD OCR MODEL ----------------

MODEL_DIR = BASE_DIR / "best_model_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


def _read_env_int(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


OCR_BATCH_SIZE = _read_env_int(
    "OCR_BATCH_SIZE",
    8 if device.type == "cuda" else 4,
)

# Lazy load model on first use
_ocr_model_cache = None
_ocr_processor_cache = None


def get_ocr_model_and_processor():
    global _ocr_model_cache, _ocr_processor_cache
    if _ocr_model_cache is None or _ocr_processor_cache is None:
        print(f"Loading OCR model from {MODEL_DIR}... (this may take a moment)")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        try:
            _ocr_processor_cache = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
            print("Processor loaded")
            model_kwargs = {"torch_dtype": torch.float16} if device.type == "cuda" else {}
            _ocr_model_cache = VisionEncoderDecoderModel.from_pretrained(
                str(MODEL_DIR),
                **model_kwargs,
            ).to(device)
            print(f"Model loaded to device: {device}")
            _ocr_model_cache.eval()
            print("OCR model loaded successfully!")
        except Exception as e:
            print(f"FAILED to load OCR model: {e}")
            import traceback

            traceback.print_exc()
            raise
    return _ocr_model_cache, _ocr_processor_cache


# ---------------- HELPERS ----------------

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def run_cropping_script():
    python_exe = BASE_DIR / "myenv" / "Scripts" / "python.exe"
    subprocess.run([str(python_exe), "extract_answer_boxes-auto.py"], check=True)


def load_labels():
    labels_path = OUTPUT_DIR / "labels.csv"
    results = []

    if not labels_path.exists():
        return results

    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                {
                    "question": int(row["question"]),
                    "raw_path": row["path"],
                }
            )

    return results


def load_answer_key(pdf_path):
    from extract_key import extract_answer_key

    return extract_answer_key(pdf_path)[1]


def build_remark(predicted, confidence, correct_answer):
    if "?" in predicted or predicted.strip() == "" or confidence < OCR_CONFIDENCE_THRESHOLD:
        return "Unable to read"
    if str(predicted) == str(correct_answer):
        return "Correct"
    return "Wrong"


def evaluate_cropped_answer(image_path, correct_answer):
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return "Unable to read", 0.0, ""

        ocr_model, processor = get_ocr_model_and_processor()
        predicted, confidence = recognize_number(
            image,
            ocr_model,
            processor,
            device,
        )

        print(
            f"  Raw prediction: '{predicted}' | confidence: {confidence:.2%} "
            f"| threshold: {OCR_CONFIDENCE_THRESHOLD}"
        )
        remark = build_remark(predicted, confidence, correct_answer)
        return remark, round(confidence * 100, 2), predicted
    except Exception as e:
        import traceback

        print(f"ERROR in evaluate_cropped_answer for {image_path}: {e}")
        traceback.print_exc()
        return "Unable to read", 0.0, ""


def evaluate_cropped_answers_batch(image_paths, correct_answers, ocr_model, processor):
    results = [("Unable to read", 0.0, "") for _ in image_paths]
    valid_indices = []
    valid_images = []

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: Unable to read cropped image: {image_path}")
            continue
        valid_indices.append(idx)
        valid_images.append(image)

    if not valid_images:
        return results

    try:
        predictions = recognize_numbers(valid_images, ocr_model, processor, device)
    except Exception as e:
        import traceback

        print(f"Batch OCR failed; falling back to single-image OCR for this batch: {e}")
        traceback.print_exc()
        predictions = []
        for image in valid_images:
            try:
                predictions.append(recognize_number(image, ocr_model, processor, device))
            except Exception:
                predictions.append(("?", 0.0))

    for result_idx, (predicted, confidence) in zip(valid_indices, predictions):
        correct_answer = correct_answers[result_idx]
        print(
            f"  Raw prediction: '{predicted}' | confidence: {confidence:.2%} "
            f"| threshold: {OCR_CONFIDENCE_THRESHOLD}"
        )
        remark = build_remark(predicted, confidence, correct_answer)
        results[result_idx] = (remark, round(confidence * 100, 2), predicted)

    return results


# ---------------- ROUTES ----------------

@app.route("/process", methods=["POST"])
def process():
    print("Received /process request")

    try:
        for f in INPUT_DIR.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass

        image_file = request.files.get("answer_sheet")
        answer_key_file = request.files.get("answer_key")

        if not image_file or image_file.filename == "":
            return {"error": "No answer sheet uploaded"}, 400

        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return {"error": "Invalid image file type"}, 400

        image_filename = secure_filename(image_file.filename)
        image_path = INPUT_DIR / image_filename
        image_file.save(image_path)
        print(f"Saved answer sheet: {image_path}")

        answer_key_dict = {}
        if answer_key_file and answer_key_file.filename != "":
            if allowed_file(answer_key_file.filename, ALLOWED_PDF_EXTENSIONS):
                pdf_filename = secure_filename(answer_key_file.filename)
                pdf_path = ANSWER_KEY_DIR / pdf_filename
                answer_key_file.save(pdf_path)
                try:
                    answer_key_dict = load_answer_key(pdf_path)
                except Exception as e:
                    print(f"Answer key load failed: {e}")
                    return {"error": str(e)}, 400
                print(f"Loaded answer key with {len(answer_key_dict)} answers")

        print("Running cropping script...")
        try:
            run_cropping_script()
            print("Cropping completed")
        except Exception as e:
            print(f"Cropping failed: {e}")
            return {"error": f"Cropping failed: {e}"}, 500

        cropped_data = load_labels()
        print(f"Loaded {len(cropped_data)} cropped cells")

        merged_results = []
        total_questions = 0
        total_correct = 0
        ocr_model = None
        processor = None

        if cropped_data:
            ocr_model, processor = get_ocr_model_and_processor()

        for batch_start in range(0, len(cropped_data), OCR_BATCH_SIZE):
            batch_items = cropped_data[batch_start : batch_start + OCR_BATCH_SIZE]
            batch_paths = [Path(item["raw_path"]) for item in batch_items]
            batch_answers = [
                answer_key_dict.get(item["question"], MISSING_ANSWER)
                for item in batch_items
            ]

            batch_first_q = batch_items[0]["question"]
            batch_last_q = batch_items[-1]["question"]
            print(
                f"Processing OCR batch {batch_start + 1}-{batch_start + len(batch_items)} "
                f"of {len(cropped_data)} | questions {batch_first_q}-{batch_last_q} "
                f"| batch_size={len(batch_items)}"
            )

            try:
                batch_results = evaluate_cropped_answers_batch(
                    batch_paths,
                    batch_answers,
                    ocr_model,
                    processor,
                )
            except Exception as e:
                print(f"Error processing OCR batch starting at question {batch_first_q}: {e}")
                import traceback

                traceback.print_exc()
                batch_results = [("Unable to read", 0.0, "") for _ in batch_items]

            for item, full_path, correct_answer, batch_result in zip(
                batch_items,
                batch_paths,
                batch_answers,
                batch_results,
            ):
                q = item["question"]
                relative_path = full_path.relative_to(OUTPUT_DIR).as_posix()
                remark, confidence, predicted = batch_result

                if correct_answer != MISSING_ANSWER:
                    total_questions += 1
                    if remark == "Correct":
                        total_correct += 1

                merged_results.append(
                    {
                        "question": q,
                        "image_url": url_for("serve_output", filename=relative_path),
                        "correct_answer": correct_answer,
                        "detected_answer": predicted,
                        "remark": remark,
                        "confidence": confidence,
                    }
                )

        accuracy = 0.0
        if total_questions > 0:
            accuracy = round((total_correct / total_questions) * 100, 2)

        csv_path = RESULTS_DIR / "evaluation_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Question",
                    "Detected Answer",
                    "Correct Answer",
                    "Remark",
                    "Confidence",
                ]
            )

            for item in merged_results:
                writer.writerow(
                    [
                        item["question"],
                        item["detected_answer"],
                        item["correct_answer"],
                        item["remark"],
                        item["confidence"],
                    ]
                )

        print(f"Processing completed. Total questions: {total_questions}, Accuracy: {accuracy}%")
        return {
            "results": merged_results,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "accuracy": accuracy,
        }

    except Exception as e:
        print(f"Error in /process: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}, 500


@app.route("/download_results")
def download_results():
    return send_from_directory(
        RESULTS_DIR,
        "evaluation_results_corrected.csv",
        as_attachment=True,
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
        writer.writerow(
            [
                "Question",
                "Detected Answer",
                "Correct Answer",
                "Remark",
                "Confidence",
            ]
        )

        for item in corrected_results:
            confidence_value = (
                "Manually Corrected"
                if item.get("manuallyEdited")
                else item["confidence"]
            )

            writer.writerow(
                [
                    item["question"],
                    item["detected_answer"],
                    item["correct_answer"],
                    item["remark"],
                    confidence_value,
                ]
            )

            if item["correct_answer"] != MISSING_ANSWER:
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
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    app.run(debug=True)
