# from ultralytics import YOLO

# model = YOLO("best.pt")

# results = model.predict(source=0, show=True, conf=0.4, save=True)


import cv2
from ultralytics import YOLO

def get_bounding_boxes():
    # Load your custom YOLO model
    model = YOLO("best.pt")  # Replace with your trained model path

    # Open video stream (0 for webcam, or replace with video path)
    video_path = 0  # Change this to "video.mp4" if using a file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video ends

        # Run YOLO inference
        results = model(frame)
        
        objects = [0, 0, 0]

        # Draw bounding boxes on the frame
        for i in range(len(results)):
            if (i > 3):
                break
            result = results[i]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class index
                
                objects[i] = {"coords": [x1, y1, x2, y2], "conf": conf, "class": model.names[cls]}
                print(f"object{i}: confidence:", conf, "class:", model.names[cls])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, model.names[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            
        cv2.imshow("YOLOv5 Object Detection", frame)
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


get_bounding_boxes()