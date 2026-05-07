#Warning: These libraries must be installed for this code to work
import os
from ultralytics import YOLO

#General
#This code is used to compare the performance of all models in one go.
#It analyzes an entire dataset that should already be annotated.
#This allows for a comparison of the results of each model with those annotated by a human.
#The tests are performed with 2 confidence levels: 1 low and 1 high.
#The results are saved in the "modelComparison.txt" file that must be saved in the same folder as this code.

#Parameters you must configure:
#1 - Path for the weights of the models. 
model1 = YOLO("Insert here the path for the weights of model 1 - best.pt")
model2 = YOLO("Insert here the path for the weights of model 2 - best.pt")
model3 = YOLO("Insert here the path for the weights of model 3 - best.pt")

#2 - Path for the dataset of valid images you want to test (images and annotations)
folder_img = 'Insert here the path to the dataset`s image folder' 
folder_txt = 'Insert here the path to the dataset`s annotation folder'

#3 - Value for confidence low (must be a number between 0 and 1)
confidence_low = 0.6 #example: 0.6

#4 - Value for confidence high (must be a number between 0 and 1. Must be higher than confidence low)
confidence_high = 0.8 #example: 0.8

with open('modelComparison.txt', 'w', encoding='utf-8') as save_file:
    save_file.write(f"") # clear file

i = 0

for file_name in os.listdir(folder_txt):
    if file_name.endswith('.txt'):
        complete_path_txt = os.path.join(folder_txt, file_name)
        complete_path_img = os.path.join(folder_img, file_name)
        complete_path_img = complete_path_img.replace('.txt', '.jpg')
        print(f"\n--- file: {file_name} has: ---")
        confirmed_total = 0
        try:
            with open(complete_path_txt, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        if (int(line[0]) == 0):
                            confirmed_total = confirmed_total + 2
                        elif (int(line[0]) == 1):
                            confirmed_total = confirmed_total + 3
                        else:
                            confirmed_total = confirmed_total + 1
                print(f"{confirmed_total} eggs")
                with open('modelComparison.txt', 'a', encoding='utf-8') as save_file:
                    save_file.write(f"file {file_name}\n")
                    save_file.write(f"{confirmed_total} eggs were counted\n")

        except Exception as e:
            print(f"Error in {file_name}: {e}")

        i = i+1

    selected_confidence = confidence_low
    results = model1(complete_path_img, conf=selected_confidence)
    predicted_total = 0
    for box in results[0].boxes:
        predicted_class = box.cls[0]
        if (predicted_class == 0):
            predicted_total = predicted_total + 2
        elif (predicted_class == 1):
            predicted_total = predicted_total + 3
        else:
            predicted_total = predicted_total + 1
    with open('modelComparison.txt','a',encoding='utf-8') as save_file:
        save_file.write(f"Model 1 identified {predicted_total} eggs with confidence {selected_confidence}\n")

    selected_confidence = confidence_high
    results = model1(complete_path_img, conf=selected_confidence)
    predicted_total = 0
    for box in results[0].boxes:
        predicted_class = box.cls[0]
        if (predicted_class == 0):
            predicted_total = predicted_total + 2
        elif (predicted_class == 1):
            predicted_total = predicted_total + 3
        else:
            predicted_total = predicted_total + 1
    with open('modelComparison.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Model 1 identified {predicted_total} eggs with confidence {selected_confidence}\n")

    selected_confidence = confidence_low
    results = model2(complete_path_img, conf=selected_confidence)
    predicted_total = 0
    for box in results[0].boxes:
        predicted_class = box.cls[0]
        if (predicted_class == 0):
            predicted_total = predicted_total + 2
        elif (predicted_class == 1):
            predicted_total = predicted_total + 3
        else:
            predicted_total = predicted_total + 1
    with open('modelComparison.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Model 2 identified {predicted_total} eggs with confidence {selected_confidence}\n")

    selected_confidence = confidence_high
    results = model2(complete_path_img, conf=selected_confidence)
    predicted_total = 0
    for box in results[0].boxes:
        predicted_class = box.cls[0]
        if (predicted_class == 0):
            predicted_total = predicted_total + 2
        elif (predicted_class == 1):
            predicted_total = predicted_total + 3
        else:
            predicted_total = predicted_total + 1
    with open('modelComparison.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Model 2 identified {predicted_total} eggs with confidence {selected_confidence}\n")

    selected_confidence = confidence_low
    results = model3(complete_path_img, conf=selected_confidence)
    predicted_total = 0
    for box in results[0].boxes:
        predicted_class = box.cls[0]
        if (predicted_class == 0):
            predicted_total = predicted_total + 2
        elif (predicted_class == 1):
            predicted_total = predicted_total + 3
        else:
            predicted_total = predicted_total + 1
    with open('modelComparison.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Model 3 identified {predicted_total} eggs with confidence {selected_confidence}\n")

    selected_confidence = confidence_high
    results = model3(complete_path_img, conf=selected_confidence)
    predicted_total = 0
    for box in results[0].boxes:
        predicted_class = box.cls[0]
        if (predicted_class == 0):
            predicted_total = predicted_total + 2
        elif (predicted_class == 1):
            predicted_total = predicted_total + 3
        else:
            predicted_total = predicted_total + 1
    with open('modelComparison.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Model 3 identified {predicted_total} eggs with confidence {selected_confidence}\n\n")
