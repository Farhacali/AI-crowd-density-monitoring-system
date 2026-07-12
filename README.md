# AI-crowd-density-monitoring-system
An AI-powered crowd monitoring system that uses YOLOv8, OpenCV, and Flask to detect, count, and monitor people in real time from images and videos. The system analyzes crowd density, generates alerts when predefined thresholds are exceeded, and provides live visualization through a web-based dashboard.

🎯 Features

-Real-time person detection using YOLOv8

-Crowd counting from images and videos

-Crowd density classification (Safe, Moderate, High, Critical)

-Adjustable density threshold

-Dashboard-based monitoring interface

-Real-time crowd statistics visualization

-Sound alarm alerts

-Mobile push notifications

-Unique person tracking in video streams

🛠 Technologies Used
-Python
-YOLOv8 (Ultralytics)
-OpenCV
-PyTorch
-Flask
-HTML/CSS
-JavaScript
-Chart.js

📂 Project Structure
AI-CROWD-DENSITY-MONITORING-SYSTEM/
│
├── app.py
├── requirements.txt
│
├── templates/
│   └── dashboard.html
│
├── static/
│   ├── screenshots/
│   └── demo_video.mp4
│
└── README.md

⚙️ How It Works
-Upload an image or video, or connect a live video feed.
-YOLOv8 detects all persons in each frame.
-The system counts detected individuals.
-Crowd density is evaluated against a predefined threshold.

If overcrowding is detected:
-Dashboard alert is displayed
-Alarm sound is triggered
-Mobile notification is sent
-Results are visualized through the monitoring dashboard.

📊 Results
Metric	    Value
Precision   87.16%
Recall	    85.82%
mAP@50	    90.19%
mAP@50-95	  76.37%

🚀 Installation
Clone the repository:
git clone https://github.com/your-username/ai-crowd-density-monitoring-system.git
cd ai-crowd-density-monitoring-system

Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Open your browser and visit:
http://127.0.0.1:5000

📸 Screenshots
Add screenshots of:
-Person detection on image
-Crowd counting on video
-Overcrowding alert
-Dashboard visualization
-Mobile notification

🔮 Future Improvements
-Crowd behavior analysis
-Violence detection
-Multi-camera tracking
-Edge device deployment
-Smart city integration
-Predictive crowd analytics

👩‍💻 Author
Farha C Ali B.Tech Artificial Intelligence & Data Science
