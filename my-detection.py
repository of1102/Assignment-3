# Author: Zian Fang
# SID: 3036457584
# Date: 10/10/2024

#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

img_path = "/home/nvidia/Desktop/banana.jpeg"
img = jetson.utils.loadImage(img_path)

detections = net.Detect(img)

for detection in detections:
    class_label = net.GetClassDesc(detection.ClassID)

    confidence = detection.Confidence

    left = detection.Left
    top = detection.Top
    right = detection.Right
    bottom = detection.Bottom

    print(f"class: {class_label}, Confidence: {confidence:.2f}")
    print(f"Coordinates: Left={left}, Top={top}, Right={right}, Bottom={bottom}")