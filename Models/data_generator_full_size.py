import os
import cv2
import math
import random
import preprocess_full_size
import numpy as np
import tensorflow as tf

SOURCE = '/visuworks/Dataset/Selected Dataset 2'
IMG_SIZE=(512, 512, 1)
OUTPUT_SIZE=(512, 512, 1)    
    
class FullSizedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, batch_size=4,
                 img_size=IMG_SIZE, output_size=OUTPUT_SIZE,
                 is_train=True, augmentation=None):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_size = output_size
        self.is_train = is_train
        self.augmentation = augmentation
        self.data = self.load_dataset()

    def load_dataset(self):
        input_path_list = []
        label_path_list = []

        for folder in os.listdir(self.dir_path):
            if "Images" in folder:
                for file in os.listdir(os.path.join(self.dir_path, folder)):
                    input_path_list.append(file)
            else:
                for file in os.listdir(os.path.join(self.dir_path, folder)):
                    label_path_list.append(file)

        input_path_list.sort()
        label_path_list.sort()
        assert len(input_path_list) == len(label_path_list)
        data = [_ for _ in zip(input_path_list, label_path_list)]

        random.shuffle(data)
        
        if self.is_train:
            return data[:-500]
        else:
            return data[-500:]
        

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        batch_data = self.data[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        inputs = np.zeros([self.batch_size, *self.img_size])
        outputs = np.zeros([self.batch_size, *self.output_size])

        for i, data in enumerate(batch_data):
            input_img_path, output_path = data
            _input = cv2.imread(os.path.join(self.dir_path, "Images", input_img_path))
            _output = cv2.imread(os.path.join(self.dir_path, "Labels", output_path))
            
            _output = cv2.cvtColor(_output, cv2.COLOR_BGR2GRAY)
            _output = ((_output == 255).astype(np.uint8) * 1).reshape(_output.shape + (1,))
                
            data = {"image": _input, "mask": _output}
            
            augmented = self.augmentation(**data)
            preprocess_full_size.apply_preprocessing(augmented)
            inputs[i] = augmented["image"]
            outputs[i] = augmented["mask"]
        return inputs, outputs