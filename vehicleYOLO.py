import numpy as np
import cv2
import pandas as pd

# Load YOLO model
weightsPath = r"C:\Users\pakhi\PycharmProjects\TrafficPred\roadTrafficForecast-master\coco\yolov3.weights"
configPath = r"C:\Users\pakhi\PycharmProjects\TrafficPred\roadTrafficForecast-master\coco\yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Load class labels
labelsPath = r"C:\Users\pakhi\PycharmProjects\TrafficPred\roadTrafficForecast-master\coco\coco.names"
try:
    with open(labelsPath) as file:
        LABELS = file.read().strip().split("\n")
except FileNotFoundError:
    print(f"Error: File '{labelsPath}' not found.")
    exit(1)

# List of vehicles to count
list_of_vehicles = ["car", "bus", "motorbike", "truck", "bicycle"]

# Initialize DataFrame to store data
data = pd.DataFrame(columns=["Frame", "Total_Vehicles", "Each_Vehicle_Count"])


# Function to perform object detection
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5 and LABELS[classID] in list_of_vehicles:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                boxes.append(box.astype("int"))
                classIDs.append(classID)

    return boxes, classIDs


# Function to get vehicle count
def get_vehicle_count(class_names):
    total_vehicle_count = 0
    dict_vehicle_count = {}
    for class_name in class_names:
        if class_name in list_of_vehicles:
            total_vehicle_count += 1
            dict_vehicle_count[class_name] = dict_vehicle_count.get(class_name, 0) + 1
    return total_vehicle_count, dict_vehicle_count


# Open video file
input_video_path = r"C:\Users\pakhi\PycharmProjects\TrafficPred\roadTrafficForecast-master\Video_Data_Samples\video1.avi"
vs = cv2.VideoCapture(input_video_path)

# Variables for object detection frequency and vehicle counting interval
object_detection_frequency = 5
vehicle_count_interval = 10
frame_count = 0

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    frame_count += 1

    # Perform object detection at specified frequency
    if frame_count % object_detection_frequency == 0:
        boxes, classIDs = detect_objects(frame)

    # Get vehicle count at specified interval
    if frame_count % vehicle_count_interval == 0:
        total_vehicles, each_vehicle = get_vehicle_count([LABELS[classID] for classID in classIDs])
        print("Total vehicles in image:", total_vehicles)
        print("Each vehicles count in image:", each_vehicle)

        # Append data to DataFrame
        data = data.append({"Frame": frame_count, "Total_Vehicles": total_vehicles, "Each_Vehicle_Count": each_vehicle},
                           ignore_index=True)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object
vs.release()
cv2.destroyAllWindows()

# Save DataFrame to a CSV file
data.to_csv("vehicle_count_data.csv", index=False)
