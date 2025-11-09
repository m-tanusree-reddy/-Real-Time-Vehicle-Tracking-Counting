# Real-Time-Vehicle-Counter_week2

This project is a full-stack web application that performs real-time vehicle detection, tracking, and counting from a video stream. It uses the YOLOv8 object detection model and the SORT tracking algorithm, serving a live dashboard through a Flask backend and a vanilla JavaScript frontend.

![app demonstration.gif](assets/demo.gif)

## Features

-   **Real-Time Object Detection**: Utilizes the powerful YOLOv8 model to detect multiple vehicle classes (car, truck, bus, motorcycle).
-   **Multi-Object Tracking**: Implements the Simple Online and Realtime Tracking (SORT) algorithm to assign and maintain a unique ID for each vehicle across frames. ByteTrack is also available as an alternative.
-   **Two-Way Line Crossing Count**: Counts vehicles moving in two directions ("Up" and "Down") as they cross a user-defined horizontal line.
-   **Live Video Streaming**: Streams the processed video feed with bounding boxes, tracker IDs, and trails to a web browser using an MJPEG stream.
-   **Interactive Web Dashboard**: A clean and responsive frontend displays the video feed and live metrics, including up/down counts, total counts, and server processing FPS.
-   **RESTful API**: The Flask backend provides API endpoints to fetch counts and performance metrics, allowing for easy integration with other services.
-   **Decoupled Architecture**: A clear separation between the Python backend (video processing) and the JavaScript frontend (user interface).
-   **User Controls**: The interface includes controls to reset counts, pause/resume the data polling, and download the count history as a CSV file.
-   **Highly Configurable**: Easily configured through environment variables to change the video source, counting line position, model, confidence thresholds, and more.

## Tech Stack

-   **Backend**: Python, Flask, Ultralytics YOLOv8, OpenCV, NumPy, SciPy, filterpy
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript
-   **Execution Environment**: Designed for easy deployment on Google Colab using Cloudflare Tunnel for public access.

## Project Structure

```
.
├── backend/
│   ├── app.py           # Main Flask application with API and video streaming
│   └── sort_tracker.py  # SORT tracking algorithm implementation
├── frontend/
│   ├── index.html       # Web page structure
│   ├── styles.css       # Styling for the web interface
│   └── script.js        # Frontend logic for API polling and UI updates
├── videos/
│   └── video.mp4 # Folder for input video files
├── vehicle-counter-h.ipynb  # Google Colab notebook for setup and execution
├── requirements.txt         # Python package dependencies
└── README.md
```

## Getting Started

There are two primary ways to run this application: using the provided Google Colab notebook (easiest) or setting it up on your local machine.

### Option 1: Quick Start with Google Colab (Recommended)

The project is designed to run out-of-the-box in a Google Colab environment.

1.  **Upload Files**: Upload the `vehicle-counter.ipynb` notebook to your Google Colab instance.
2.  **Prepare Video**: Upload your desired video file to your Google Drive.
3.  **Mount Drive**: Run the cell in the notebook to mount your Google Drive.
4.  **Configure**: In the "Configure the Application" cell, update the `VIDEO_PATH` to point to your video file in Google Drive.
    ```python
    # Example path
    os.environ["VIDEO_PATH"] = "/content/drive/MyDrive/videos/traffic.mp4"
    ```
5.  **Adjust Settings**: In the same cell, you can adjust the counting line's vertical position (`LINE_Y`), the tracker mode (`TRACKER_MODE`), and other parameters as needed.
6.  **Run All**: Click `Runtime > Run all` to execute all the cells in the notebook. This will install dependencies, set up the project, and start the backend and frontend servers.
7.  **Access the App**: The final cell will generate and display a styled, clickable button. Click this button to open your live application in a new browser tab.

### Option 2: Running on a Local Machine

#### 1. Create `requirements.txt` File
Before you begin, create a file named `requirements.txt` in the root of your project directory and add the following content:
```
# requirements.txt
ultralytics==8.2.28
flask==3.0.3
flask-cors==4.0.1
filterpy==1.4.5
opencv-python-headless==4.9.0.80
scipy==1.13.1
numpy==1.26.4
```

#### 2. Clone Repository & Setup Environment
**Clone the repository**:
```bash
git clone [https://github.com/your-username/Real-Time-Vehicle-Counter.git](https://github.com/your-username/Real-Time-Vehicle-Counter.git)
cd Real-Time-Vehicle-Counter
```

**Prerequisites and Installation:**

1.  **Python Version**: This project is tested and fully compatible with **Python 3.11.9**. We strongly recommend using `pyenv` to manage Python versions.
    ```bash
    # Install Python 3.11.9 using pyenv
    pyenv install 3.11.9
    # Set it as the local version for your project
    pyenv local 3.11.9
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    # Create the virtual environment
    python3 -m venv venv
    # Activate it (on macOS/Linux)
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```
    Your terminal prompt should now show `(venv)`.

3.  **Install Required Libraries**:
    Inside your activated virtual environment, install the packages from the file you created:
    ```bash
    pip install -r requirements.txt
    ```

#### 3. Configure the Application
Place your video file in the `videos/` folder. Then, set the required environment variables in your terminal.
```bash
# For Linux/macOS
export VIDEO_PATH="videos/sample_traffic.mp4"
export LINE_Y="360"

# For Windows (Command Prompt)
set VIDEO_PATH="videos\sample_traffic.mp4"
set LINE_Y="360"
```

#### 4. Run the Servers
**Run the Backend Server** (in your first terminal):
```bash
cd backend
python app.py
```
The backend will run at `http://localhost:5000`.

**Run the Frontend Server** (in a *second* terminal):
```bash
cd frontend
python -m http.server 8000
```
The frontend will be available at `http://localhost:8000`.

**Connect Frontend to Backend**:
Open your browser to the following URL:
`http://localhost:8000/?backend=http://localhost:5000`

### 🔧 Troubleshooting Library Installation

If `pip install -r requirements.txt` fails, it's likely due to a system-specific conflict. If this occurs, you can allow `pip` to resolve the dependencies for you.

**Follow this fallback process inside your activated virtual environment:**

1.  **Attempt to Install Without Strict Versions:**
    ```bash
    pip install ultralytics "flask>=3.0" flask-cors filterpy opencv-python-headless scipy numpy
    ```
2.  **Generate a New `requirements.txt`:**
    If the command above succeeds and the application runs correctly, lock in your working library versions by creating a new `requirements.txt` file.
    ```bash
    # This overwrites the old file with your new, working versions
    pip freeze > requirements.txt
    ```

## Configuration

The application's behavior can be customized via environment variables.

| Variable           | Description                                                                                             | Default      |
| ------------------ | ------------------------------------------------------------------------------------------------------- | ------------ |
| `VIDEO_PATH`       | **(Required)** The full path to the input video file.                                                   | -            |
| `LINE_Y`           | The vertical pixel position of the horizontal counting line.                                            | `360`        |
| `TRACKER_MODE`     | The tracking algorithm to use. Options: `"SORT"` or `"BYTE"`.                                           | `SORT`       |
| `RESIZE_WIDTH`     | Resizes the frame to this width before processing. Smaller values (e.g., 640) increase speed.           | `960`        |
| `MODEL_PATH`       | The YOLOv8 model file to use (e.g., `yolov8n.pt`). It will be downloaded if not found.      | `yolov8s.pt` |
| `MODEL_DEVICE`     | The device to run the model on. Options: `"auto"`, `"cuda"`, `"cpu"`.                                   | `auto`       |
| `CONF_THRESH`      | The confidence threshold for object detection.                                        | `0.35`       |
| `IOU_THRESH`       | The Intersection over Union (IoU) threshold for NMS and tracker association.            | `0.50`       |
| `RESET_ON_LOOP`    | If `"true"`, resets counts every time the video loops. If `"false"`, counts are cumulative.               | `false`      |
| `TRAIL_LEN`        | The number of recent center points to store for drawing an object's trail.                              | `20`         |


## How It Works

1.  **Video Ingestion**: The Flask backend uses OpenCV to read the input video file frame by frame.
2.  **Detection**: Each frame is passed to the loaded YOLOv8 model, which returns a list of bounding boxes for detected vehicles.
3.  **Filtering**: Detections are filtered based on the confidence threshold and whether they belong to the target vehicle classes.
4.  **Tracking**: The filtered bounding boxes are passed to the `Sort` tracker. The tracker uses a Kalman Filter to predict movement and the Hungarian algorithm to associate detections with existing tracks, assigning a consistent ID to each vehicle.
5.  **Line Crossing Logic**: The application checks the center coordinate of each tracked bounding box. If a vehicle's center crosses the pre-defined `LINE_Y` position between frames, the corresponding "Up" or "Down" counter is incremented.
6.  **API and Streaming**: The processed frame, annotated with bounding boxes and track IDs, is encoded as a JPEG and sent via an MJPEG stream on the `/video_feed` endpoint. A separate REST API provides JSON data for counts and performance metrics.
7.  **Frontend Display**: The JavaScript frontend displays the MJPEG stream in an `<img>` tag and uses `fetch` to periodically poll the API endpoints, updating the metrics dashboard in real-time.

## API Endpoints

The Flask server exposes the following endpoints:

-   `GET /video_feed`: Serves the MJPEG stream of the processed video.
-   `GET /api/counts`: Returns the current vehicle counts.
    -   **Response**: `{"up": 10, "down": 8}`
-   `GET /api/metrics`: Provides performance and configuration metrics.
    -   **Response**: `{"fps": 25.5, "model_device": "cuda", ...}`
-   `GET /api/reset`: Resets the vehicle counts and the tracker's state.
    -   **Response**: `{"status": "ok", "message": "Counts and state reset."}`
-   `GET /health`: A simple health check endpoint.
    -   **Response**: `{"ok": true, "service": "vehicle-counter", ...}`
## 📧 Contact

*M Tanusree Reddy*  
📍 BE AIDS @ CMRIT  
📧 [m.tanusreereddy@gmail.com]
