#Warning: These libraries must be installed for this code to work
from ultralytics import YOLO

#General
#This code is very simple and used to analyze a single image 
#using one of the models. The result is not saved and is displayed 
#through a print function showing the total encountered and the 
#amount of eggs in each layer.

#Parameters you must configure:
#1 - Path for the weights of the model. 
model = YOLO("Insert here the path for the weights of the model - best.pt") 

#2 - Path for the image you want to analyze
image = "insert path for the image file - image.jpg"

#3 - Value for confidence (must be a number between 0 and 1)
selected_confidence = 0.5 #Example: 0.5

results = model(image, conf=selected_confidence)
results[0].show()

total = 0
predicted_1_egg = 0
predicted_2_egg = 0
predicted_3_egg = 0

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = box.conf[0]
    egg_class = box.cls[0]
    if(egg_class == 0):
        total = total + 2
        predicted_2_egg = predicted_2_egg + 1
    elif(egg_class == 1):
        total = total + 3
        predicted_3_egg = predicted_3_egg + 1
    else:
        total = total + 1
        predicted_1_egg = predicted_1_egg + 1

print(f"Total number of eggs found: {int(total)}\n")
print(f"1_Egg: {predicted_1_egg}; 2_Eggs: {predicted_2_egg}; 3_Eggs {predicted_3_egg}")