import os
import random
import shutil

data_dir = 'data'
train_images_dir = os.path.join(data_dir, 'training_images')
val_images_dir = os.path.join(data_dir, 'validation_images')

if not os.path.exists(val_images_dir):
    os.makedirs(val_images_dir)

all_images = [img for img in os.listdir(train_images_dir) if img.endswith('.jpg')]
random.shuffle(all_images)

split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

def move_file(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)

for img in val_images:
    img_src = os.path.join(train_images_dir, img)
    img_dst = os.path.join(val_images_dir, img)
    move_file(img_src, img_dst)

    annotation_file = img.replace('.jpg', '.txt')
    annotation_src = os.path.join(train_images_dir, annotation_file)
    annotation_dst = os.path.join(val_images_dir, annotation_file)
    if os.path.exists(annotation_src):
        move_file(annotation_src, annotation_dst)

print(f'Total images: {len(all_images)}')
print(f'Training images: {len(train_images)}')
print(f'Validation images: {len(val_images)}')
