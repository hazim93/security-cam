import cv2

def main():
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam live feed started. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # If frame is read correctly ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            
            # Display the resulting frame
            cv2.imshow('Webcam Live Feed', frame)
            
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