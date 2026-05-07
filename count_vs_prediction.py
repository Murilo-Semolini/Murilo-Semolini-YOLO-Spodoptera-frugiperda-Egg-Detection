#Warning: These libraries must be installed for this code to work
import os
from ultralytics import YOLO

#General
#This code is used to test a single model and verify if the results 
#predicted by the model are close to the results counted by a human.
#The results are stored in a file "count_vs_prediction.txt" 
#which must be saved in the same folder as the code.

#Parameters you must configure:
#1 - Path for the weights of the model. 
model = YOLO("Insert here the path for the weights of the model - best.pt") 

#2 - Path for the dataset of valid images you want to test (images and annotations)
folder_img = 'Insert here the path to the dataset`s image folder'
folder_txt = 'Insert here the path to the dataset`s annotation folder' 

#3 - Value for confidence (must be a number between 0 and 1)
selected_confidence = 0.5 #Example: 0.5

with open('count_vs_prediction.txt', 'w', encoding='utf-8') as save_file:
    save_file.write(f"") 

for file_name in os.listdir(folder_txt):
    if file_name.endswith('.txt'):
        complete_path_txt = os.path.join(folder_txt, file_name)
        complete_path_img = os.path.join(folder_img, file_name)
        complete_path_img = complete_path_img.replace('.txt', '.jpg')
        print(f"\n--- file: {file_name} has: ---")
        confirmed_total = 0
        confirmed_1_egg = 0
        confirmed_2_egg = 0
        confirmed_3_egg = 0
        try:
            with open(complete_path_txt, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip(): 
                        if (int(line[0]) == 0):
                            confirmed_total = confirmed_total + 2
                            confirmed_2_egg = confirmed_2_egg + 1
                        elif (int(line[0]) == 1):
                            confirmed_total = confirmed_total + 3
                            confirmed_3_egg = confirmed_3_egg + 1
                        else:
                            confirmed_total = confirmed_total + 1
                            confirmed_1_egg = confirmed_1_egg + 1
                print(f"{confirmed_total} eggs")
                with open('count_vs_prediction.txt', 'a', encoding='utf-8') as save_file:
                    save_file.write(f"file {file_name}\n")
                    save_file.write(f"{confirmed_total} eggs were counted\n")
                    save_file.write(f"1_Egg: {confirmed_1_egg}; 2_Eggs: {confirmed_2_egg}; 3_Eggs {confirmed_3_egg}\n")

        except Exception as e:
            print(f"Error in {file_name}: {e}")

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
    with open('count_vs_prediction.txt','a',encoding='utf-8') as save_file:
        save_file.write(f"Model identified {predicted_total} eggs with confidence {selected_confidence}\n")
        save_file.write(f"1_Egg: {predicted_1_egg}; 2_Eggs: {predicted_2_egg}; 3_Eggs {predicted_3_egg}\n\n")