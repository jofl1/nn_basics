from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8m.pt')  

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame)
    
    # Draw results on frame
    annotated_frame = results[0].plot()
    
    # Show result
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()