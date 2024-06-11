import pandas as pd
import os

data_dir = 'data'
train_images_dir = os.path.join(data_dir, 'training_images')
val_images_dir = os.path.join(data_dir, 'validation_images')
bbox_df = pd.read_csv(os.path.join(data_dir, 'train_solution_bounding_boxes (1).csv'))

def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def is_bbox_valid(bbox):
    return all(0 <= coord <= 1 for coord in bbox)

for index, row in bbox_df.iterrows():
    img_name = row['image']
    bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
    img_path = os.path.join(train_images_dir, img_name)

    if os.path.exists(img_path):
        img_size = (640, 480)  # Adjust this to your actual image size
        yolo_bbox = convert_bbox_to_yolo(img_size, bbox)
        
        if is_bbox_valid(yolo_bbox):
            yolo_label = f'0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n'
            annotation_file = os.path.join(train_images_dir, img_name.replace('.jpg', '.txt'))
            with open(annotation_file, 'a') as f:
                f.write(yolo_label)
        else:
            print(f'Ignoring invalid bounding box in {img_name}: {yolo_bbox}')
