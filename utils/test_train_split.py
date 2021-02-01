import os
import glob
import ntpath
import numpy as np

aug_img_dir ='./data/bars/augmented'

augmented_jpgs = list(glob.iglob(os.path.join(aug_img_dir, '*.jpg')))
aug_jpg_paths = dict()
for jpg_aug_name in augmented_jpgs:
    jpg_name = ntpath.basename(os.path.normpath(''.join(jpg_aug_name.split('__')[:-1]) + '.jpg'))
    if jpg_name not in aug_jpg_paths:
        aug_jpg_paths[jpg_name] = []
    aug_jpg_paths[jpg_name].append(ntpath.basename(jpg_aug_name))


label_file = './data/labels.txt'
train_label_file = './data/train_labels.txt'
test_label_file = './data/test_labels.txt'

label_files = {
    'train': [],
    'test': []
}

with open(label_file, 'r+') as out:
    lines = out.readlines()
    for line in lines:
        if len(line.strip()):
            s = line.split(' ')
            jpg_name = ' '.join(s[:-1])
            labelValue = s[-1]
            labelValue = labelValue.replace('\n', '')
            
            set_type = 'test' if np.random.rand() > 0.829 else 'train'

            label_files[set_type].append(line)

            for aug_img in aug_jpg_paths[jpg_name]:
                aug_line = f'{os.path.join("augmented", aug_img)} {labelValue}\n'
                label_files[set_type].append(aug_line)

with open(train_label_file, 'w+') as f:
    f.writelines(label_files['train'])
    f.close()

with open(test_label_file, 'w+') as f:
    f.writelines(label_files['test'])
    f.close()

# train_imgs = []
# test_imgs = []

# test_imgs_names = set()