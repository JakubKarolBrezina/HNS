# YOLO Object Detection

This project demonstrates object detection using YOLO (You Only Look Once) with pretrained models provided by Ultralytics. YOLO is a real-time object detection system capable of detecting multiple objects in images, videos, and live camera streams without the need to train a custom model.

The project runs on Python 3.9 or newer and uses pip as the package manager. Required Python libraries are ultralytics, opencv-python, and numpy.

To set up the project, first clone the repository and navigate into its directory:

git clone https://github.com/your-username/yolo-object-detection.git  
cd yolo-object-detection

It is recommended to create a virtual environment to avoid dependency conflicts:

python -m venv venv

Activate the virtual environment.

On Windows:
venv\Scripts\activate

On macOS or Linux:
source venv/bin/activate

After activating the environment, install all required dependencies:

pip install --upgrade pip  
pip install ultralytics opencv-python numpy

Once the dependencies are installed, the project is ready to run. Object detection can be performed on images, videos, or in real time using a webcam.

To run object detection on an image:

python detect.py --source data/image.jpg

To run object detection on a video:

python detect.py --source data/video.mp4

To run real-time object detection using the default webcam:

python detect.py --source 0

During the first execution, the pretrained YOLO model will be downloaded automatically, which may take a short while depending on the internet connection.

While the program is running, detected objects are displayed with bounding boxes, class labels, and confidence scores. In webcam mode, the results are shown live in a window. Press the Q key to close the window and stop the program.

The output of the detection is automatically saved in the runs/detect/ directory. Each execution creates a new subfolder (for example exp, exp2, etc.) containing the processed images or videos with detected objects highlighted.

The expected output consists of images or videos where objects such as people, cars, animals, or other recognizable items are detected and visually marked using bounding boxes and labels along with confidence values.

Supported image formats include JPG and PNG. Supported video formats include MP4 and AVI. The project uses pretrained YOLO models such as yolov8n.pt or yolov8s.pt by default and does not require any additional training.

This project is intended for educational and demonstration purposes.

