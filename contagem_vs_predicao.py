#Este código serve para testar um único modelo e verificar os resultados encontrados pelo
#modelo são próximos dos resultados encontrados por um ser humano.
#Os resultados são armazenados em um arquivo "contagem_vs_predicao.txt" que deve ser salvo
#na mesma pasta do código

import os
from ultralytics import YOLO

#troque o caminho nesta linha pelo caminho dos pesos do modelo escolhido no seu computador
model = YOLO("C:/Users/IFSP SRT/PycharmProjects/muriloIC/runs/detect/train/weights/best.pt") 

folder_txt = 'datasetV11/valid/labels' #troque para o caminho do seu dataset
folder_img = 'datasetV11/valid/images' #troque para o caminho do seu dataset

with open('contagem_vs_predicao.txt', 'w', encoding='utf-8') as save_file:
    save_file.write(f"") # clear file

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
                    if line.strip(): # linha não está vazia
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
                with open('contagem_vs_predicao.txt', 'a', encoding='utf-8') as save_file:
                    save_file.write(f"arquivo {file_name}\n")
                    save_file.write(f"Foram contados {confirmed_total} ovos\n")
                    save_file.write(f"1_Egg: {confirmed_1_egg}; 2_Eggs: {confirmed_2_egg}; 3_Eggs {confirmed_3_egg}\n")

        except Exception as e:
            print(f"Error in {file_name}: {e}")

    selected_confidence = 0.5 #Selecione a confiança que você quer para o teste
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
    print(f"The model identified: {predicted_total} eggs")
    with open('contagem_vs_predicao.txt','a',encoding='utf-8') as save_file:
        save_file.write(f"Modelo identificou {predicted_total} ovos com confiança {selected_confidence}\n")
        save_file.write(f"1_Egg: {predicted_1_egg}; 2_Eggs: {predicted_2_egg}; 3_Eggs {predicted_3_egg}\n\n")