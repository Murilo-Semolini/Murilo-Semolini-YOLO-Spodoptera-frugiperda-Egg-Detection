import os
from ultralytics import YOLO

model1 = YOLO("C:/Users/IFSP SRT/PycharmProjects/muriloIC/runs/detect/train2/weights/best.pt")
model3 = YOLO("C:/Users/IFSP SRT/PycharmProjects/muriloIC/runs/detect/train3/weights/best.pt")
model2 = YOLO("C:/Users/IFSP SRT/PycharmProjects/muriloIC/runs/detect/train7/weights/best.pt")

folder_txt = 'datasetV11/valid/labels'
folder_img = 'datasetV11/valid/images'

with open('comparacao.txt', 'w', encoding='utf-8') as save_file:
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
                    if line.strip(): # line is not empty
                        if (int(line[0]) == 0):
                            confirmed_total = confirmed_total + 2
                        elif (int(line[0]) == 1):
                            confirmed_total = confirmed_total + 3
                        else:
                            confirmed_total = confirmed_total + 1
                print(f"{confirmed_total} eggs")
                with open('comparacao.txt', 'a', encoding='utf-8') as save_file:
                    save_file.write(f"arquivo {file_name}\n")
                    save_file.write(f"Foram contados {confirmed_total} ovos\n")

        except Exception as e:
            print(f"Error in {file_name}: {e}")

        i = i+1

    selected_confidence = 0.7
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
    with open('comparacao.txt','a',encoding='utf-8') as save_file:
        save_file.write(f"Modelo 1 identificou {predicted_total} ovos com confiança {selected_confidence}\n")

    selected_confidence = 0.8
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
    with open('comparacao.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Modelo 1 identificou {predicted_total} ovos com confiança {selected_confidence}\n")

    selected_confidence = 0.7
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
    with open('comparacao.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Modelo 2 identificou {predicted_total} ovos com confiança {selected_confidence}\n")

    selected_confidence = 0.8
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
    with open('comparacao.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Modelo 2 identificou {predicted_total} ovos com confiança {selected_confidence}\n")

    selected_confidence = 0.7
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
    with open('comparacao.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Modelo 3 identificou {predicted_total} ovos com confiança {selected_confidence}\n")

    selected_confidence = 0.8
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
    with open('comparacao.txt', 'a', encoding='utf-8') as save_file:
        save_file.write(f"Modelo 3 identificou {predicted_total} ovos com confiança {selected_confidence}\n\n")