# Spodoptera frugiperda Egg Detection with YOLO

## Project Overview

This project uses a YOLO-based neural network to identify *Spodoptera frugiperda* eggs in microscope images.
The models are capable of:

* Detecting egg clusters using bounding boxes
* Identifying the egg layer in the image
* Estimating the total number of eggs present

The repository contains:

* Three trained YOLO models (`model_1`, `model_2`, `model_3`)
* Test and evaluation scripts
* The image dataset used during training and validation
* Training configuration files and parameters

Among the available models, `model_3` achieved the best overall performance.

---

# Dataset

The dataset is divided into two main folders:

* Training images
* Validation images

Each folder contains:

* A folder with the `.jpg` images
* A folder with `.txt` annotation files

The annotation files contain:

* Bounding box coordinates
* The corresponding class for each box

---

## Bounding Box Classes

| Class Index | Class Name | Description       |
| ----------- | ---------- | ----------------- |
| 0           | `2_Eggs`   | Middle layer eggs |
| 1           | `3_Eggs`   | Top layer eggs    |
| 2           | `1_Egg`    | Bottom layer eggs |

The class numbering may appear unintuitive due to an annotation mistake made during the labeling process. However, all scripts and models were adapted to correctly handle these class indexes.

---

## Layer Interpretation

The images were captured using a microscope with illumination coming from below the sample platform.

Because of this lighting setup:

* `1_Egg` represents eggs located on the bottom layer. These eggs appear brighter because the light passes through them more easily.
* `2_Eggs` represents eggs located in the middle layer. These eggs appear darker because the light must first pass through the bottom layer.
* `3_Eggs` represents eggs located on the top layer. These eggs appear even darker because the light must pass through the two lower layers before reaching them.

---

# Egg Counting Method

To validate the predicted counts, the scripts first analyze the annotation `.txt` files before processing the image with the neural network.

Using the annotated bounding boxes and their classes, the scripts estimate the number of eggs counted by a human annotator.

The total number of eggs is calculated as follows:

* `3_Eggs` boxes are multiplied by 3
* `2_Eggs` boxes are multiplied by 2
* `1_Egg` boxes are added directly without multiplication

This approach assumes:

* A `3_Eggs` detection represents one visible egg with two additional eggs below it
* A `2_Eggs` detection represents one visible egg with one additional egg below it

---

# Scripts

## `count_vs_prediction`

Tests a single model and compares the predicted egg count with the count obtained from the annotation files.

The results are saved in:

```text
count_vs_prediction.txt
```

This file must be located in the same folder as the script.

---

## `modelComparison`

Compares the performance of all trained models at once.

The script:

* Analyzes an entire annotated image folder
* Compares model predictions with human annotations
* Performs tests using two confidence levels:

  * One low confidence threshold
  * One high confidence threshold

The results are saved in:

```text
modelComparison.txt
```

This file must be located in the same folder as the script.

---

## `test_1Model_MultipleImagens`

Analyzes an entire folder of images using a single model.

The results are displayed through `print()` statements and include:

* Total detected eggs
* Number of eggs detected in each layer

The results are not saved to a file.

---

## `test_1Model_1Image`

Analyzes a single image using one of the trained models.

The results are displayed through `print()` statements and include:

* Total detected eggs
* Number of eggs detected in each layer

The results are not saved to a file.

---

# Training

The training process was made through the files:

* `mainCode`
* `config.yaml`

These files contain the base code used to start and configure the YOLO training process, but for each training new parameters were set. The list of alterations made for each model can be foun in `train parameters.txt`.

---