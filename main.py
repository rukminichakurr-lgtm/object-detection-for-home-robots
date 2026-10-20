from ultralytics import YOLO
from pathlib import Path

# Load the pretrained object detection model
model = YOLO("yolov8n.pt")


def detect_objects(image_path):
    """Detect objects in an image and display the results."""

    image = Path(image_path)

    if not image.exists():
        print(f"Image not found: {image_path}")
        return

    results = model.predict(
        source=str(image),
        conf=0.40,
        save=True
    )

    detected_objects = []

    for result in results:
        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                object_name = names[class_id]

                detected_objects.append(object_name)

                print(
                    f"Object: {object_name} | "
                    f"Confidence: {confidence:.2%}"
                )

    print("\nTotal objects detected:", len(detected_objects))

    print("\nObject counts:")
    for name in sorted(set(detected_objects)):
        print(f"{name}: {detected_objects.count(name)}")


if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip()
    detect_objects(image_path)