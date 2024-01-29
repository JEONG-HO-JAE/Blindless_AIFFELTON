import os
import cv2
import math
import random
import numpy as np
import tensorflow as tf
import preprocess
  
class AdaptiveDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, number_of_images = 4,
                 img_size=(256, 256, 1), output_size=(256, 256, 1),
                 is_train=True, is_test=False, augmentation=None):
        self.number_of_images = number_of_images
        self.dir_path = dir_path
        self.batch_size = number_of_images * (512//img_size[0]) * (512//img_size[0])
        self.img_size = img_size
        self.output_size = output_size
        self.is_train = is_train
        self.is_test = is_test
        self.augmentation = augmentation
        self.data = self.load_dataset()
  
    def load_dataset(self):
        input_path_list = []
        label_path_list = []

        for folder in os.listdir(self.dir_path):
            if "Images" in folder:
                for file in os.listdir(os.path.join(self.dir_path, folder)):
                    input_path_list.append(os.path.join(self.dir_path, folder, file))
            else:
                for file in os.listdir(os.path.join(self.dir_path, folder)):
                    label_path_list.append(os.path.join(self.dir_path, folder, file))

        input_path_list.sort()
        label_path_list.sort()
        assert len(input_path_list) == len(label_path_list)
        
        data = [_ for _ in zip(input_path_list, label_path_list)] 
        random.shuffle(data)
        
        if self.is_test:
            return data
        
        if self.is_train:
            return data[:-500]
        else:
            return data[-500:]
    
    def __len__(self):
        return math.ceil(len(self.data) / (self.number_of_images))
    
    def  __getitem__(self, index):
        batch_data = self.data[
            index * (self.number_of_images):
            (index + 1) * (self.number_of_images)
        ]
        
        image_patches = []
        label_patches = []
        
        for image_path, label_path in batch_data:
            image = cv2.imread(image_path)
            image = preprocess.apply_cutomized_preprocess(image)
            _label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = ((_label == 255).astype(np.uint8) * 1)
        
            # 이미지 크기에 따라 조절 및 패치 추출
            if max(image.shape) < 1024:
                patches_image, patches_label = preprocess.resize_and_extract_patches(image, label, self.img_size[0], 512)
            elif max(image.shape) < 2048:
                patches_image, patches_label = preprocess.resize_and_extract_patches(image, label, self.img_size[0],1024)
            else:
                patches_image, patches_label = preprocess.resize_and_extract_patches(image, label, self.img_size[0], 2048)

            # Accumulate patches from all images in the batch
            image_patches.extend(patches_image)
            label_patches.extend(patches_label)
        
        # Choose a fixed number of patches per image
        # print(len(image_patches),self.batch_size)
        indices = random.sample(range(len(image_patches)), self.batch_size)
        selected_image_patches = [image_patches[i] for i in indices]
        selected_label_patches = [label_patches[i] for i in indices]
            
        # Convert TensorFlow tensors to numpy arrays
        selected_image_patches = np.array(selected_image_patches)
        selected_label_patches = np.array(selected_label_patches)
 
        augmented_data = {"image": selected_image_patches,
                          "mask": selected_label_patches}     
        augmented_data = self.augmentation(**augmented_data)
        
        input_patches, label_patches = augmented_data['image'], augmented_data['mask']
            
        return input_patches, label_patches
    
class FullSizedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, batch_size=4,
                 img_size=(512, 512, 1), output_size=(512, 512, 1),
                 is_train=True, is_test=False, augmentation=None):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_size = output_size
        self.is_train = is_train
        self.is_test = is_test
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
        
        if self.is_test:
            return data
        
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
            _input =  preprocess.apply_cutomized_preprocess(_input)
            _output = cv2.imread(os.path.join(self.dir_path, "Labels", output_path))
            _output = cv2.cvtColor(_output, cv2.COLOR_BGR2GRAY)
            _output = ((_output == 255).astype(np.uint8) * 1).reshape(_output.shape + (1,))
                
            data = {"image": _input, "mask": _output}
            
            augmented = self.augmentation(**data)
            
            inputs[i] = augmented["image"].reshape((self.img_size[0], self.img_size[0], 1))
            outputs[i] = augmented["mask"].reshape((self.img_size[0], self.img_size[0], 1))
        return inputs, outputs
    
class SlicedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, number_of_images = 2,
                 img_size=(512, 512, 1), output_size=(512, 512, 1), resize_shape = (1024, 1024, 1),
                 is_train=True, is_test=False, augmentation=None):
        self.dir_path = dir_path
        self.resize_shape = resize_shape
        self.number_of_images = number_of_images
        self.batch_size =  (resize_shape[0] // img_size[0]) *  (resize_shape[0] // img_size[0]) * number_of_images
        self.img_size = img_size
        self.output_size = output_size
        self.is_train = is_train
        self.is_test = is_test
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
        
        if self.is_test:
            return data
        
        if self.is_train:
            return data[:-500]
        else:
            return data[-500:]
        

    def __len__(self):
        return math.ceil(len(self.data) / (self.number_of_images))

    def __getitem__(self, index):
        batch_data = self.data[
            index * (self.number_of_images):
            (index + 1) * (self.number_of_images)
        ]

        inputs = np.zeros([self.batch_size, *self.img_size])
        outputs = np.zeros([self.batch_size, *self.output_size])
        
        for i in range(0, self.number_of_images):
            # print("index:", index)
            # print("len(batch_data):", len(batch_data))
            # print("i:", i)
            input_img_path, output_path = batch_data[i]
            _input = cv2.imread(os.path.join(self.dir_path, "Images", input_img_path))
            _input =  preprocess.apply_cutomized_preprocess(_input)
            _output = cv2.imread(os.path.join(self.dir_path, "Labels", output_path))
            _output = cv2.cvtColor(_output, cv2.COLOR_BGR2GRAY)
            _output = ((_output == 255).astype(np.uint8) * 1).reshape(_output.shape + (1,))
            
            data = {"image": _input, "mask": _output}  
            augmented = self.augmentation(**data)
            
            patch_index_per_one_image = self.batch_size // self.number_of_images
            number_of_row = self.resize_shape[0] // self.img_size[0]
            for p in range(0, patch_index_per_one_image):
                k = (i * patch_index_per_one_image) + p
                start_row, end_row = self.img_size[0] * (p // number_of_row), self.img_size[0] * (p // number_of_row + 1) 
                start_col, end_col = self.img_size[0] * (p % number_of_row), self.img_size[0] * (p % number_of_row + 1)
                # print(start_row, end_row, start_col, end_col)
                # print(p)
                inputs[k] = augmented["image"][start_row:end_row, start_col:end_col].reshape((self.img_size[0], self.img_size[0], 1))
                outputs[k] = augmented["mask"][start_row:end_row, start_col:end_col].reshape((self.img_size[0], self.img_size[0], 1))
        # print(len(inputs), len(outputs))
        return inputs, outputs
    
class RandomlyCropGeneraor(tf.keras.utils.Sequence):
    def __init__(self, dir_path, number_of_images = 2,
                 input_size=(128, 128, 1),
                 is_train=True, is_test=False, augmentation=None):
        self.dir_path = dir_path
        self.number_of_images = number_of_images
        self.input_size = input_size
        self.batch_size =  2^21 // input_size[0] // input_size[0]
        self.is_train = is_train
        self.is_test = is_test
        self.augmentation = augmentation
        self.data = self.load_dataset()
        
    def load_dataset(self):
        input_path_list = []
        label_path_list = []

        for folder in os.listdir(self.dir_path):
            if "Images" in folder:
                for file in os.listdir(os.path.join(self.dir_path, folder)):
                    input_path_list.append(os.path.join(self.dir_path, folder, file))
            else:
                for file in os.listdir(os.path.join(self.dir_path, folder)):
                    label_path_list.append(os.path.join(self.dir_path, folder, file))

        input_path_list.sort()
        label_path_list.sort()
        assert len(input_path_list) == len(label_path_list)
            
        data = [_ for _ in zip(input_path_list, label_path_list)] 
        random.shuffle(data)
        
        if self.is_test:
            return data
        
        if self.is_train:
            return data[:-500]
        else:
            return data[-500:]
            
    def __len__(self):
        return math.ceil(len(self.data) / (self.number_of_images))

    def __getitem__(self, index):
        batch_data = self.data[
            index * (self.number_of_images):
            (index + 1) * (self.number_of_images)
        ]

        image_patches = []
        label_patches = []
            
        for image_path, label_path in batch_data:
            image = cv2.imread(image_path)
            image = preprocess.apply_cutomized_preprocess(image)
            _label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = ((_label == 255).astype(np.uint8) * 1)
                
            # 이미지 크기에 따라 조절 및 패치 추출
            patches_image, patches_label = preprocess.extract_patches(image, label, self.input_size[0])
                
            image_patches.extend(patches_image)
            label_patches.extend(patches_label)
            
        indices = random.sample(range(len(image_patches)), self.batch_size)
        selected_image_patches = [image_patches[i] for i in indices]
        selected_label_patches = [label_patches[i] for i in indices]
                
        # Convert TensorFlow tensors to numpy arrays
        selected_image_patches = np.array(selected_image_patches)
        selected_label_patches = np.array(selected_label_patches)
    
        augmented_data = {"image": selected_image_patches,
                         "mask": selected_label_patches}     
        augmented_data = self.augmentation(**augmented_data)
            
        input_patches, label_patches = augmented_data['image'], augmented_data['mask']
                
        return input_patches, label_patches