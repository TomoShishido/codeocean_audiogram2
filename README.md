# codeocean_audiogram2

For code ocean capsule reproduction of "Audiogram Detection AI Trained Using Size-Fixed Frame and Bounding-Boxes"

'''
This capsule is provided to reproduce the results of the manuscript entitled
"Audiogram Detection AI Trained Using Size-Fixed Frame and Bounding-Boxes"

Datasets:
data/datasets/training_datasets directory contains the following training/validation datasets: 1. DatasetA (rect416 images) for DETR or YOLO training 2. DatasetB (original images) for DETR training 3. DatasetC (original images) for YOLO training

data/datasets/test_datasets directory contains the following test datasets (in YOLO format): 1. Original test dataset 2. Original test dataset without photo images 3. Rect416 test dataset

Models:
data/models/ 1. rect416DETR_audiogramdetr_checkpoint0499.pth (trained with datasetA)
classes: a0 (right air-conduction threshold), a1 (right bone-conduction threshold),
a2 (left air-conduction threshold), a3 (left bone-conduction threshold),
and a4 (overlapping right and left air-conduction thresholds). 2. rect416YOLO_audiogramdetr2yolo_best_20241023.pt (trained with datasetA)
classes: the above a0-a4 3. originalDETR_checkpoint0499.pth (trained with datasetB)
classes: aa0 (audiogram frame), aa1 (0-dB line),
and aa2–aa6 (corresponding to a0–a4, respectively) 4. (conventional model)originalYOLO_aagraphmodel3_ntnshladded_aa6_best_20240315.pt (trained with datasetC)
classes: the above aa0-aa6
Utilities
General: code/utilities/general 1. extract_resize2rect416image.py: to extract a rectangle from each original image and resize it to 416 x 416 pixcels 2. drawingfullgraph_rect416.py: to redraw a standardized audiogram for a rect416 format 3. confusionmatrix_testdataset.py: to depict a confusion matrix and calculate model performance

    df_preps: code/utilities/df_preps
    to prepare a DataFrame (df) from YOLOv5x detection results in respective Figures

DETR: code/utilities/DETR/ 1. rect416DETRprediction1.py: to perform inference using rect416DETR model

YOLOv5
YOLOv5 is included in this capsule : License is the same as the YOLOv5 repository.
https://github.com/ultralytics/yolov5
GNU Affero General Public License v3.0
DETR
DETR is included in this capsule : License is the same as the original DETR repository.
https://github.com/facebookresearch/detr
Apache License Version 2.0

Others:
The license used for each model is the same as the original license.
