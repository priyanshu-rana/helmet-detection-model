from ultralytics import YOLO
import cv2
import numpy as np

def main():
    # Load your trained model
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting live helmet detection...")
    print("Press 'q' to quit")
    print("Press 'c' to change confidence threshold")
    
    # Start with higher confidence threshold
    conf_threshold = 0.5
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from camera")
            break
        
        # Run detection on the frame with higher confidence
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        
        # Get the annotated frame with bounding boxes
        annotated_frame = results[0].plot()
        
        # Add text overlay with detection info and confidence threshold
        detections = results[0]
        if detections.boxes is not None:
            for i, box in enumerate(detections.boxes):
                # Get class name and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                # Add text to frame
                text = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated_frame, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show current confidence threshold
        cv2.putText(annotated_frame, f"Confidence: {conf_threshold}", (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the result
        cv2.imshow("Helmet Detection - Live", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Change confidence threshold
            conf_threshold = 0.7 if conf_threshold == 0.5 else 0.5
            print(f"Confidence threshold changed to: {conf_threshold}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main() 