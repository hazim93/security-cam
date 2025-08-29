import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response

# Define security-relevant classes (based on COCO dataset class indices)
security_class_names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    15: 'cat',
    16: 'dog'
}

# Initialize Flask app
app = Flask(__name__)

# Global variables for camera and model
cap = None
model = None

def initialize_camera_and_model():
    """Initialize the webcam and YOLO model"""
    global cap, model
    
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Load the YOLO model
    try:
        model = YOLO("yolo11n.pt")  # Using the nano version for faster inference
        print("YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure you have internet connection for automatic model download")
        cap.release()
        return False

def generate_frames():
    """Generate video frames for streaming"""
    global cap, model
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        # Run YOLO object detection on the frame with filtered classes
        try:
            if model is not None:
                results = model(frame, classes=list(security_class_names.keys()))
                
                # Plot the results on the frame
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
                
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            
            # Convert to bytes and yield
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        except Exception as e:
            print(f"Error during object detection: {e}")
            # Encode the original frame if detection fails
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    """Main function to start the Flask app"""
    print("Initializing camera and model...")
    if not initialize_camera_and_model():
        return
    
    print("Detecting only security-relevant objects:", list(security_class_names.values()))
    print("Starting Flask server...")
    print("Open your browser and go to http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        # Clean up resources
        if cap is not None:
            cap.release()

if __name__ == "__main__":
    main()