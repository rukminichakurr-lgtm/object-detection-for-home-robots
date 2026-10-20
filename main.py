from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import argparse
import json

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.40
OUTPUT_DIR = Path("outputs")


def detect_objects(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        print("Error: Image file not found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)

    result = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD
    )[0]

    output_image = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"
    result.save(filename=str(output_image))

    report = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        report.append({
            "object": result.names[class_id],
            "confidence": round(float(box.conf[0]), 4),
            "position": [x1, y1, x2, y2]
        })

    report_path = OUTPUT_DIR / f"{image_path.stem}_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n===== OBJECT DETECTION REPORT =====")

    for name, count in Counter(
        item["object"] for item in report
    ).items():
        print(f"{name}: {count}")

    print("\nTotal objects detected:", len(report))
    print("Detected image saved:", output_image)
    print("JSON report saved:", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Input image path")
    args = parser.parse_args()

    image_path = args.image or input("Enter image path: ").strip()
    detect_objects(image_path)


