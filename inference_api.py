from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=""
)

# Read the image
image_path = "IMG_4915.JPG"
image = cv2.imread(image_path)

# infer on a local image
result = CLIENT.infer(image_path, model_id="elevator-dbzwt/1")

# Draw bounding boxes for each prediction
for prediction in result["predictions"]:
    # Extract coordinates and dimensions
    x = int(prediction["x"])
    y = int(prediction["y"])
    width = int(prediction["width"])
    height = int(prediction["height"])
    
    # Calculate the top-left and bottom-right corners
    x1 = int(x - width/2)
    y1 = int(y - height/2)
    x2 = int(x + width/2)
    y2 = int(y + height/2)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add label with confidence
    label = f"{prediction['class']}: {prediction['confidence']:.2f}"
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the annotated image
output_path = "annotated_" + image_path
cv2.imwrite(output_path, image)
print(f"Annotated image saved as: {output_path}")
