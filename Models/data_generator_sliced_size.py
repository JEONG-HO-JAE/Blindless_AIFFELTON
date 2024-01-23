import os
import cv2
import math
import random
import preprocess_slice_size
import numpy as np
import tensorflow as tf

SOURCE = '/visuworks/Dataset/Selected Dataset 2'
  
class SlicedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, batch_size=128,
                 img_size=(64, 64, 1), output_size=(64, 64, 1),
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
        return math.ceil(len(self.data) / (self.batch_size//16))

    def __getitem__(self, index):
        batch_data = self.data[
            index * (self.batch_size//16):
            (index + 1) * (self.batch_size//16)
        ]

        inputs = np.zeros([self.batch_size, *self.img_size])
        outputs = np.zeros([self.batch_size, *self.output_size])
        
        for i in range(min(len(batch_data), 8)):
            input_img_path, output_path = batch_data[i]
            _input = cv2.imread(os.path.join(self.dir_path, "Images", input_img_path))
            _output = cv2.imread(os.path.join(self.dir_path, "Labels", output_path))

            _output = cv2.cvtColor(_output, cv2.COLOR_BGR2GRAY)
            _output = ((_output == 255).astype(np.uint8) * 1).reshape(_output.shape + (1,))
                
            data = {"image": _input, "mask": _output}  
            augmented = self.augmentation(**data)
            preprocess_slice_size.apply_preprocessing(augmented)
            
            for j in range(0, 16):
                k = (i * 16) + j
                start_row, end_row = 64 * (j // 4), 64 * (j // 4 + 1) 
                start_col, end_col = 64 * (j % 4), 64 * (j % 4 + 1)
                inputs[k] = augmented["image"][start_row:end_row, start_col:end_col]
                outputs[k] = augmented["mask"][start_row:end_row, start_col:end_col]
        return inputs, outputs