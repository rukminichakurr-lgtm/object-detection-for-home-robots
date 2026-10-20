
from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import json

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.40
OUTPUT_DIR = Path("outputs")
IMAGE_PATH = OUTPUT_DIR / "detected_objects.jpg"

OUTPUT_DIR.mkdir(exist_ok=True)

def detect_objects(image_path):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(MODEL_NAME)
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD
    )

    result = results[0]
    result.save(filename=str(OUTPUT_DIR / "detected_objects.jpg"))

    report = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        report.append({
            "object": result.names[class_id],
            "confidence": round(float(box.conf[0]), 4),
            "position": [x1, y1, x2, y2]
        })

    with open(OUTPUT_DIR / "detection_report.json", "w") as file:
        json.dump(report, file, indent=4)

    print("\n===== OBJECT DETECTION REPORT =====")
    for name, count in Counter(x["object"] for x in report).items():
        print(f"{name}: {count}")

    print("\nTotal objects detected:", len(report))
    print("Report saved:", OUTPUT_DIR / "detection_report.json")
    print("Image saved:", OUTPUT_DIR / "detected_objects.jpg")

if __name__ == "__main__":
    detect_objects(IMAGE_PATH)
