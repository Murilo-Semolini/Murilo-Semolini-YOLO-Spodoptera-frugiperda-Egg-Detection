#Warning: These libraries must be installed for this code to work
import os
from ultralytics import YOLO
from pathlib import Path

#General
#This code is used to compare the performance of all models in one go.
#It analyzes an entire  image folder that should already be annotated.
#This allows for a comparison of the results of each model with those annotated by a human.
#The tests are performed with multiple confidence levels and the results are saved in the 
#"allModelsComparison.txt" file that must be saved in the same folder as this code.

#Parameters you must configure:
#1 - Path for the folder of the models. 
models_folder = Path(r"path to the models folder")

#2 - Path for the dataset of valid images you want to test (images and annotations)
folder_img = 'path to the images folder' 
folder_txt = 'path to the labels folder'

# 3 - Confidence values. They must be between 0 and 1. The values here are an example
confidences = [0.4, 0.5, 0.6, 0.7, 0.8]

# ==================================================

# Making a list with the models paths
# This will result in models[0], models[1], models[2], etc.

models = []

for model_dir in models_folder.iterdir():
    if model_dir.is_dir():
        best_pt = model_dir / "weights" / "best.pt"

        if best_pt.exists():
            models.append(YOLO(str(best_pt)))

# ==================================================

# functions to count the eggs from the boxes and from the annotation file

def count_eggs_from_boxes(boxes):
    total = 0

    for box in boxes:

        predicted_class = int(box.cls[0])

        if predicted_class == 0:
            total += 2

        elif predicted_class == 1:
            total += 3

        else:
            total += 1

    return total


def count_eggs_from_annotation(annotation_path):

    total = 0

    with open(annotation_path, "r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            class_id = int(line[0])

            if class_id == 0:
                total += 2

            elif class_id == 1:
                total += 3

            else:
                total += 1

    return total

# ==================================================

# Clear output file

with open("allModelsComparison.txt", "w", encoding="utf-8") as save_file:
    pass

# ==================================================

# Main loop

for file_name in os.listdir(folder_txt):

    if not file_name.endswith(".txt"):
        continue

    complete_path_txt = os.path.join(folder_txt, file_name)

    complete_path_img = os.path.join(
        folder_img,
        file_name.replace(".txt", ".jpg")
    )

    print(f"\n--- file: {file_name} ---")

    try:

        confirmed_total = count_eggs_from_annotation(
            complete_path_txt
        )

        print(f"{confirmed_total} eggs")

        with open("allModelsComparison.txt", "a", encoding="utf-8") as save_file:

            save_file.write(f"file {file_name}\n")
            save_file.write(
                f"{confirmed_total} eggs were counted\n"
            )

    except Exception as e:

        print(f"Error in {file_name}: {e}")
        continue

    # ==============================================

    # Test all models
    
    for model_number, model in enumerate(models, start=1):

        for confidence in confidences:

            try:

                results = model(
                    complete_path_img,
                    conf=confidence
                )

                predicted_total = count_eggs_from_boxes(
                    results[0].boxes
                )

                print(
                    f"Model {model_number} | "
                    f"Conf {confidence} | "
                    f"Eggs: {predicted_total}"
                )

                with open(
                    "allModelsComparison.txt",
                    "a",
                    encoding="utf-8"
                ) as save_file:

                    save_file.write(
                        f"Model {model_number} identified "
                        f"{predicted_total} eggs "
                        f"with confidence {confidence}\n"
                    )

            except Exception as e:

                print(
                    f"Error testing model "
                    f"{model_number}: {e}"
                )

    with open(
        "allModelsComparison.txt",
        "a",
        encoding="utf-8"
    ) as save_file:

        save_file.write("\n")