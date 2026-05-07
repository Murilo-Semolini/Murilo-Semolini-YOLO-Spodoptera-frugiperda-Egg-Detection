#Warning: These libraries must be installed for this code to work
import os
from ultralytics import YOLO

#General
#This code serves to analyze an entire folder of images using one 
# of the models. The result is not saved and is displayed through 
# a print function showing the total encountered and the amount of 
# eggs in each layer.

#Parameters you must configure:
#1 - Path for the weights of the model. 
model = YOLO("Insert here the path for the weights of the model - best.pt") 

#2 - Path for the folder of images you want to analyze 
folder_img = 'Insert here the path to the image folder'

#3 - Value for confidence (must be a number between 0 and 1)
selected_confidence = 0.5 #Example: 0.5

current_image = 0
for file_name in os.listdir(folder_img):
    if file_name.endswith('.jpg'):
        complete_path_img = os.path.join(folder_img, file_name)

    results = model(complete_path_img, conf=selected_confidence)
    predicted_total = 0
    predicted_1_egg = 0
    predicted_2_egg = 0
    predicted_3_egg = 0
    for box in results[0].boxes:
        predicted_class = box.cls[0]
        if (predicted_class == 0):
            predicted_total = predicted_total + 2
            predicted_2_egg = predicted_2_egg + 1
        elif (predicted_class == 1):
            predicted_total = predicted_total + 3
            predicted_3_egg = predicted_3_egg + 1
        else:
            predicted_total = predicted_total + 1
            predicted_1_egg = predicted_1_egg + 1
    
    print(f"Total number of eggs found on image {int(current_image)}: {int(predicted_total)}\n")
    print(f"1_Egg: {predicted_1_egg}; 2_Eggs: {predicted_2_egg}; 3_Eggs {predicted_3_egg}\n\n")
    
    current_image = current_image + 1