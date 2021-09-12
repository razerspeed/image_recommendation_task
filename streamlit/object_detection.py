import torchvision
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
transform_test = transforms.Compose([

    transforms.Resize((512, 512)),
    transforms.ToTensor()

])

def object_detections(image):
    img1 = transform_test(image)
    model.eval()
    predictions = model(img1.unsqueeze(0))
    threshold = .8
    img2 = F.to_pil_image(img1)
    draw = ImageDraw.Draw(img2)
    count_label = []
    for i in range(len(predictions[0]['labels'])):
        if predictions[0]['scores'][i] > threshold:
            boxes = predictions[0]['boxes'][i]
            img_boxes = boxes.to(torch.int64).tolist()
            draw.rectangle(img_boxes, width=4, outline="red")
            label = COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'][i]]
            draw.text((img_boxes[0], img_boxes[1]), label)
            count_label.append(label)

    (unique, counts) = np.unique(count_label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    return img2,frequencies