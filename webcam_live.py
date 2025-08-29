import cv2
from ultralytics import YOLO

def main():
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
    
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Load the YOLO model
    try:
        model = YOLO("yolo11n.pt")  # Using the nano version for faster inference
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure you have internet connection for automatic model download")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print("Webcam live feed with object detection started. Press 'q' to quit.")
    print("Detecting only security-relevant objects:", list(security_class_names.values()))
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # If frame is read correctly ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            
            # Run YOLO object detection on the frame with filtered classes
            try:
                results = model(frame, classes=list(security_class_names.keys()))
                
                # Plot the results on the frame
                annotated_frame = results[0].plot()
                
                # Display the resulting frame with object detection
                cv2.imshow('Webcam Live Feed with Object Detection', annotated_frame)
                
            except Exception as e:
                print(f"Error during object detection: {e}")
                # Display the original frame if detection fails
                cv2.imshow('Webcam Live Feed with Object Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()