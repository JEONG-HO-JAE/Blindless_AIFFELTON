from tensorflow.keras import backend as K
from keras.optimizers import *
from sklearn.metrics import confusion_matrix
import os
import cv2
import preprocess
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def accuracy(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))

    return (true_positives + true_negatives) / (possible_positives + possible_negatives + K.epsilon())

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

def iou(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    return true_positives / ( true_positives + false_positives + false_negatives)


def retrieve_evaluation_value(model, generator):
    # Evaluate the model on the test data
    evaluation = model.evaluate(generator)

    # Extract individual metric values from the evaluation result
    loss_value = evaluation[0]
    sensitivity_value = evaluation[1]
    specificity_value = evaluation[2]
    accuracy_value = evaluation[3]
    f1_value = evaluation[4]
    iou_value = evaluation[5]
    
    return loss_value, sensitivity_value, specificity_value, accuracy_value, f1_value, iou_value

def retrive_all_evaluation_of_test_dataset(generator, path, custom_objects):
    list = os.listdir(path)
    list.sort()
    def numerical_sort(value):
        parts = value.split("-")
        try:
            return int(parts[0])
        except ValueError:
            return value
    sorted_list = sorted(list, key=numerical_sort)
    
    loss = []
    sen = []
    spe = []
    acc = []
    f1 = []
    iou = []
    
    for i, model_path in enumerate (sorted_list):
        model_path = os.path.join(path, model_path)
        print(f"{i + 1}번째 epoch 결과")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=custom_objects['DiceLoss'], 
              metrics=[custom_objects['sensitivity'], custom_objects['specificity'], custom_objects['accuracy'], custom_objects['f1_score'],  custom_objects['iou']])

        loss_value, sensitivity_value, specificity_value, accuracy_value, f1_value, iou_value = retrieve_evaluation_value(model, generator)
        loss.append(loss_value)
        sen.append(sensitivity_value)
        spe.append(specificity_value)
        acc.append(accuracy_value)
        f1.append(f1_value)
        iou.append(iou_value)
    
    return loss, sen, spe, acc, f1, iou

def testdataset_retrieve_path_list(path, img_name):
    input_path_list = []
    label_path_list = []
    file_list = []
    
    for img_path in os.listdir(os.path.join(path, "Images")):
        if img_name in img_path:
            file_list.append(img_path)
            
    for file in file_list:
        for label_file in os.listdir(os.path.join(path, "Labels")):
            if label_file[:-3] == file[:-3]:
                input_path_list.append(os.path.join(path, "Images", file))
                label_path_list.append(os.path.join(path, "Labels", label_file))
    
    return input_path_list, label_path_list

def retrieve_FG(model, preproc_img, label_img, img_size, preproc):
    data = {"image": preproc_img, "mask": label_img}
    processed = preproc(**data)
    
    inputs = np.zeros([1, *img_size])
    inputs[0] = np.expand_dims(processed["image"], axis=-1)
    output = model(inputs)
    output = (output[0].numpy()>0.5).astype(np.uint8).squeeze(-1)
    return processed["mask"].squeeze(-1), output 

def retrieve_SG(model, org_img, img_size, resize_shape, preproc):
    preproc_img = preprocess.apply_cutomized_preprocess(org_img)
    
    data = preproc(**data)
    
    patch_index_per_one_image = (resize_shape[0] // img_size[0]) * (resize_shape[0] // img_size[0])
    inputs = np.zeros([patch_index_per_one_image, *img_size])
    outputs = np.zeros([patch_index_per_one_image, *img_size])
    output = np.zeros(resize_shape, dtype=np.uint8)           
       
    number_of_row = resize_shape[0] // img_size[0] 
    for p in range(0, patch_index_per_one_image):
        start_row, end_row = img_size[0] * (p // number_of_row), img_size[0] * (p // number_of_row + 1) 
        start_col, end_col = img_size[0] * (p % number_of_row), img_size[0] * (p % number_of_row + 1)
        inputs[p] = np.expand_dims(data["image"][start_row:end_row, start_col:end_col], axis=-1)
   
    outputs= model(inputs)
        
    for p in range(0, patch_index_per_one_image):
        start_row, end_row = img_size[0] * (p // number_of_row), img_size[0] * (p // number_of_row + 1) 
        start_col, end_col = img_size[0] * (p % number_of_row), img_size[0] * (p % number_of_row + 1)
        # print(start_row, end_col, start_col, end_col)
        outputs_np = (outputs[p].numpy()>0.5).astype(np.uint8)*255
        # print(outputs_np.shape)
        output[start_row:end_row, start_col:end_col] = outputs_np
        
    return preproc_img, output  

def testdataset_result(path, img_name, data_generator, model, img_size, preproc):
    input_path_list, label_path_list = testdataset_retrieve_path_list(path, img_name)
    label_imgs = []
    output_imgs = []
    
    for i, img_path in enumerate(input_path_list):
        org_img = cv2.imread(img_path)
        preproc_img = preprocess.apply_cutomized_preprocess(org_img)
        label_img = cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE)
        label_img = ((label_img == 255).astype(np.uint8) * 1).reshape(label_img.shape + (1,))
    
        if data_generator == 'FG':
            label, output = retrieve_FG(model, preproc_img, label_img, img_size, preproc)
        else:
            label, output  = retrieve_SG(model, preproc_img, img_size, preproc)
        label_imgs.append(label)
        output_imgs.append(output)
    return input_path_list, output_imgs, label_imgs

def evaluate_testdataset_result(path, img_name, data_generator, model, img_size, preproc):
    input_path_list, output_imgs, label_imgs = testdataset_result(path, img_name, data_generator, model, img_size, preproc)
    for i, input_path in enumerate(input_path_list):
        file = os.path.basename(input_path)
        tn, fp, fn, tp = confusion_matrix(label_imgs[i].flatten(), output_imgs[i].flatten()).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        iou = tp / (tp + fp + fn)
        print(f"====={file} result======")
        print(f"sensitivity: {sen}")
        print(f"specificity: {spe}")
        print(f"IOU: {iou}")

def plot_result(path, img_name, model, img_size, preproc):
    input_path_list, label_path_list = testdataset_retrieve_path_list(path, img_name)
    label_imgs = []
    preproc_imgs = []
    output_imgs = []
    
    for i, img_path in enumerate(input_path_list):
        org_img = cv2.imread(img_path)
        preproc_img = preprocess.apply_cutomized_preprocess(org_img)
        label_img = cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE)
        label_img = ((label_img == 255).astype(np.uint8) * 1).reshape(label_img.shape + (1,))
    
        data = {"image": preproc_img, "mask": label_img}
        processed = preproc(**data)
        
        inputs = np.zeros([1, *img_size])
        inputs[0] = np.expand_dims(processed["image"], axis=-1)
        output = model(inputs)
        output = (output[0].numpy()>0.5).astype(np.uint8).squeeze(-1)
        
        preproc_imgs.append(processed["image"])
        label_imgs.append(processed["mask"])
        output_imgs.append(output)
    
        plt.figure(figsize=(20, 20))
        file = os.path.basename(input_path_list[i])
        tn, fp, fn, tp = confusion_matrix(label_imgs[i].flatten(), output_imgs[i].flatten()).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        iou = tp / (tp + fp + fn)
        print(f"====={file} result======")
        print(f"sensitivity: {sen}")
        print(f"specificity: {spe}")
        print(f"IOU: {iou}")
        #Preprocessed Image
        plt.subplot(1, 3, 1)
        plt.imshow(processed["image"], cmap='gray')
        plt.title(os.path.basename(input_path_list[i]))
        
        plt.subplot(1, 3, 2)
        plt.imshow(processed["mask"], cmap='gray')
        plt.title('Label image')
        
        # Label Image
        plt.subplot(1, 3, 3)
        plt.imshow(output,  cmap='gray')
        plt.title('Predicted Image')
        
        plt.show()

def plot_result_for_SG(path, img_name, model, img_size, resize_shape, preproc):
    input_path_list, label_path_list = testdataset_retrieve_path_list(path, img_name)
    label_imgs = []
    preproc_imgs = []
    output_imgs = []
    
    for i, img_path in enumerate(input_path_list):
        org_img = cv2.imread(img_path)
        preproc_img = preprocess.apply_cutomized_preprocess(org_img)
        label_img = cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE)
        label_img = ((label_img == 255).astype(np.uint8) * 1).reshape(label_img.shape + (1,))
    
        data = {"image": preproc_img, "mask": label_img}
        data = preproc(**data)
        patch_index_per_one_image = (resize_shape[0] // img_size[0]) * (resize_shape[0] // img_size[0])
        inputs = np.zeros([patch_index_per_one_image, *img_size])
        outputs = np.zeros([patch_index_per_one_image, *img_size])
        output = np.zeros(resize_shape, dtype=np.uint8)          
        
        number_of_row = resize_shape[0] // img_size[0] 
        for p in range(0, patch_index_per_one_image):
            start_row, end_row = img_size[0] * (p // number_of_row), img_size[0] * (p // number_of_row + 1) 
            start_col, end_col = img_size[0] * (p % number_of_row), img_size[0] * (p % number_of_row + 1)
            inputs[p] = np.expand_dims(data["image"][start_row:end_row, start_col:end_col], axis=-1)
                
        outputs= model(inputs)
            
        for p in range(0, patch_index_per_one_image):
            start_row, end_row = img_size[0] * (p // number_of_row), img_size[0] * (p // number_of_row + 1) 
            start_col, end_col = img_size[0] * (p % number_of_row), img_size[0] * (p % number_of_row + 1)
            # print(start_row, end_col, start_col, end_col)
            outputs_np = (outputs[p].numpy()>0.5).astype(np.uint8)*255
            # print(outputs_np.shape)
            output[start_row:end_row, start_col:end_col] = outputs_np
        
        preproc_imgs.append(data["image"])
        label_img = data["mask"]
        label_imgs.append(label_img)
        output_imgs.append(output)
        
        plt.figure(figsize=(20, 20))
        file = os.path.basename(input_path_list[i])
        conf_matrix = confusion_matrix(label_imgs[i].squeeze(-1).flatten(), (output_imgs[i].squeeze(-1)/255).flatten())
        tn, fp, fn, tp = conf_matrix.ravel()

        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        iou = tp / (tp + fp + fn)
        print(f"====={file} result======")
        print(f"sensitivity: {sen}")
        print(f"specificity: {spe}")
        print(f"IOU: {iou}")
        #Preprocessed Image
        plt.subplot(1, 3, 1)
        plt.imshow(data["image"], cmap='gray')
        plt.title(os.path.basename(input_path_list[i]))
        
        plt.subplot(1, 3, 2)
        plt.imshow(data["mask"], cmap='gray')
        plt.title('Label image')
        
        # Label Image
        plt.subplot(1, 3, 3)
        plt.imshow(output,  cmap='gray')
        plt.title('Predicted Image')
        
        plt.show()
        
def plot_result_for_AG(path, img_name, model, img_size):
    input_path_list, label_path_list = testdataset_retrieve_path_list(path, img_name)

    for i, img_path in enumerate(input_path_list):
        file = os.path.basename(img_path)
        org_img = cv2.imread(img_path)
        
        label_img = cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE)
        label_img = ((label_img == 255).astype(np.uint8) * 1).reshape(label_img.shape + (1,))
        
        preproc_img = preprocess.apply_cutomized_preprocess(org_img)
        if max(org_img.shape) < 1024:
            preproc_img = cv2.resize(preproc_img, (512, 512))
        elif max(org_img.shape) < 2048:
            preproc_img = cv2.resize(preproc_img, (1024, 1024))
        else:
            preproc_img = cv2.resize(preproc_img, (2048, 2048))
        
        preproc_img = preproc_img[:, :, np.newaxis]
        index = (preproc_img.shape[0] // img_size[0]) * (preproc_img.shape[0] // img_size[0])
        input_size=img_size
        full_img_size=(preproc_img.shape[0], preproc_img.shape[0], 1)
            
        inputs = np.zeros([index, *input_size])
        outputs = np.zeros([index, *input_size])
        output = np.zeros(full_img_size, dtype=np.uint8)
        
        data = {"image": preproc_img}
        j = preproc_img.shape[0]//img_size[0]
        for i in range(0, index):
            start_row = img_size[0] * (i // j)
            end_row = img_size[0] * (i // j + 1) 
            start_col = img_size[0] * (i % j)
            end_col = img_size[0] * (i % j + 1)
            inputs[i] = data["image"][start_row:end_row, start_col:end_col]
            
        outputs= model(inputs)
            
        for i in range(0, index):
            start_row = img_size[0] * (i // j)
            end_row = img_size[0] * (i // j + 1) 
            start_col = img_size[0] * (i % j)
            end_col = img_size[0] * (i % j + 1)
            outputs_np = (outputs[i].numpy()>0.5).astype(np.uint8)*255
            output[start_row:end_row, start_col:end_col] = outputs_np
        
        label_img = cv2.resize(label_img, (output.shape[1], output.shape[0]))
        plt.figure(figsize=(20, 20))
        
        conf_matrix = confusion_matrix(label_img.flatten(), (output.squeeze(-1)/255).flatten())
        tn, fp, fn, tp = conf_matrix.ravel()

        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        iou = tp / (tp + fp + fn)
        print(f"====={file} result======")
        print(f"sensitivity: {sen}")
        print(f"specificity: {spe}")
        print(f"IOU: {iou}")
        #Preprocessed Image
        plt.subplot(1, 3, 1)
        plt.imshow(data["image"], cmap='gray')
        plt.title(file)
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.resize(label_img, (data["image"].shape[1], data["image"].shape[0])), cmap='gray')
        plt.title('Label image')
        
        # Label Image
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.resize(output, (data["image"].shape[1], data["image"].shape[0])),  cmap='gray')
        plt.title('Predicted Image')
        
        plt.show()
        

def plot_test_evaluation_result(loss, sen, spe, acc, f1, iou):
    # Plotting
    epochs = range(1, len(loss) + 1)
    
    # Loss plot
    plt.plot(epochs, loss, "r", label="Loss")
    plt.title("Loss")
    plt.legend()
    plt.figure()
    
    # Sensitivity and Specificity plot
    plt.plot(epochs, sen, "g", label="Sensitivity")
    plt.plot(epochs, spe, "b", label="Specificity")
    plt.title("Sensitivity & Specificity")
    # Set y-axis range
    plt.ylim(0.6, 1.0)
    # Adjust y-axis ticks spacing
    # Set y-axis ticks with 0.01 intervals
    plt.yticks(np.arange(0.6, 1.01, 0.02))
    plt.figure()
    
    # Accuracy plot
    plt.plot(epochs, acc, "r", label="Accuracy")
    plt.plot(epochs, f1, "g", label="F1 Score")
    plt.plot(epochs, iou, "b", label="IoU")
    plt.title("Accuracy & F1 Score & IoU")
    plt.legend()
    plt.figure()
    
    plt.show()