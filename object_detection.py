#0bject Detection on Image
# #Import All the Required Libraries
from ultralytics import YOLO

#Load the YOL011 Pre-Trained Model
model = YOLO("yolo11n.pt")
#Perform Object Detection on an Image
results = model.predict("resources/images/baggage_claim.PNG", save=True)

