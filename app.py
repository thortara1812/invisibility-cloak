from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folders for uploaded and processed videos
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function to check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Invisibility Cloak Processing Function
def process_invisibility_cloak(input_path, output_path):
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # ✨ Set to CPU (Important change)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Open uploaded video
    cap = cv2.VideoCapture(input_path)
    ret, background = cap.read()
    if not ret:
        print("Error: Cannot read video or first frame.")
        cap.release()
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up video output for live streaming
    def generate_frames():
        # Prepare grayscale background for optical flow
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        # HSV mask range for white/grey detection
        lower_whitegrey = np.array([0, 0, 130])
        upper_whitegrey = np.array([180, 50, 255])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optical Flow Alignment
            flow = cv2.calcOpticalFlowFarneback(background_gray, frame_gray, None,
                                                0.5, 5, 25, 5, 7, 1.5, 0)
            y, x = np.mgrid[0:frame_height, 0:frame_width].astype(np.float32)
            remap_x = x + flow[..., 0]
            remap_y = y + flow[..., 1]
            aligned_bg = cv2.remap(background, remap_x, remap_y, cv2.INTER_LINEAR)

            # YOLO Object Detection
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes = []
            confidences = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 0 and confidence > 0.5:  # Person class
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        w = int(detection[2] * frame_width)
                        h = int(detection[3] * frame_height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)
                    roi = frame[y:y + h, x:x + w]
                    if roi.size == 0:
                        continue

                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower_whitegrey, upper_whitegrey)
                    ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])

                    # Cloak if mostly white/grey
                    if ratio > 0.5:
                        frame[y:y + h, x:x + w] = aligned_bg[y:y + h, x:x + w]

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file part"
    
    file = request.files['video']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return redirect(url_for('show_video', filename=filename))
    else:
        return "Invalid file type."

@app.route('/show_video/<filename>')
def show_video(filename):
    return render_template('index.html', filename=filename)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return process_invisibility_cloak(filepath, None)  # No output path, we are streaming live

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
