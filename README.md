
# Object Detection for Home Robots

An AI-powered object detection system built using Python and YOLOv8 to identify objects in images, estimate their locations, and generate detection reports.

## Features

- Detect multiple objects in images
- Display object names and confidence scores
- Identify object positions using bounding boxes
- Count detected objects
- Save images with detection annotations
- Generate JSON detection reports
- Explore object detection for home robotics

## Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV
- Pillow
- Matplotlib
- JSON

##  Installation

Install Python, then install the required libraries:

```bash
pip install -r requirements.txt
🚀 How to Run
Run the program:
python main.py
When prompted, enter the path to your image.
Example:
Enter image path: sample.jpg
Make sure the image file exists at the specified path.
📊 Output
The project generates:
An annotated image with detected objects
A JSON report containing object names, confidence scores, and bounding box coordinates
A summary showing the number of detected objects
Output files are saved in the outputs folder.
🧠 AI Model
This project uses the pretrained YOLOv8n model. It can recognize supported object categories from its training dataset.
Detecting additional objects, such as pens, may require custom model training.
🔮 Future Improvements
Real-time object detection using a camera
Custom training for additional household objects
Improved obstacle detection for home robots
Integration with a robot navigation system
⚠️ Disclaimer
This is an educational project. It is not a safety-certified robot navigation system.

### Step 14: Save karo
1. **Commit changes** par tap karo.
2. Commit message likho: `Improve project documentation`
3. Dobara **Commit changes** par tap karo.

**Important:** README mein run karne ka example tabhi kaam karega jab tum image ka sahi path dogi.

Ho jaaye toh mujhe **“Done”** batao. Phir hum project ko actual image ke saath test karenge. 🚀
