import torch
from torch import nn
import torchvision.transforms as T
import time
import os, glob
from PIL import Image

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = [
  'a0','a1','a2','a3','a4'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

# Convert boxes to YOLO format (x_center, y_center, width, height)
def convert_to_yolo_format(boxes, image_size):
    img_w, img_h = image_size
    yolo_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        yolo_boxes.append([x_center, y_center, width, height])
    return yolo_boxes

# Save the results in YOLO format
def save_yolo_results(image_name, probas, boxes, output_dir,image_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    yolo_boxes = convert_to_yolo_format(boxes, image_size)
    yolo_file_path = os.path.join(output_dir, f"{image_name.split('.')[0]}.txt")

    with open(yolo_file_path, 'w') as f:
        for p, box in zip(probas, yolo_boxes):
            cl = p.argmax()
            label = CLASSES[cl]
            confidence = p[cl].item()
            if confidence > 0.5:  # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                f.write(f"{cl} {box[0]:.5f} {box[1]:.5f} {box[2]:.5f} {box[3]:.5f}\n")

# model load
model = torch.hub.load('facebookresearch/detr',
                      'detr_resnet50',
                      pretrained=False,
                      num_classes=5)
model = model.to(device)  # Move model to the device

checkpoint = torch.load('/Users/tesutoyoukanrisha/慈恵データ/fastaudiogramdetection/weights/audiogramdetr_checkpoint0499.pth', map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

def predict_detr_testdata(IMAGE_FILE_PATH,OUTPUT_DIR):
    time_sta = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = glob.glob(IMAGE_FILE_PATH)
        #To skip .txt files
    for file in files:
        if file.endswith('jpg') or file.endswith('JPG'):    
            image_name = os.path.basename(file)

            im = Image.open(file).convert('RGB')
            img = transform(im).unsqueeze(0).to(device)  # Move input data to the device

            # propagate through the model
            # time_sta = time.time()
            outputs = model(img)
            # time_end = time.time()
            # tim = time_end - time_sta
            # print("time =", tim)

            # keep only predictions with 0.7+ confidence
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.5 #revised from 0.3 to 0.5

            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

            # Save detection results in YOLO format
            save_yolo_results(image_name, probas[keep], bboxes_scaled.tolist(), OUTPUT_DIR, im.size)
