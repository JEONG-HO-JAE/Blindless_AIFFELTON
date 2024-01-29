import cv2
import json, os, random
import preprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def retrieve_path_list(SOURCE, num_images_to_select):
    input_path_list = []
    label_path_list = []

    file_list = random.sample(os.listdir(os.path.join(SOURCE, "Images")), num_images_to_select)
    for file in file_list:
        for label_file in os.listdir(os.path.join(SOURCE, "Labels")):
            if label_file[:-3] == file[:-3]:
                input_path_list.append(os.path.join(SOURCE, "Images", file))
                label_path_list.append(os.path.join(SOURCE, "Labels", label_file))
    return input_path_list, label_path_list

def retrieve_FG_model_result(model, org_img, img_size, preproc):
    preproc_img = preprocess.apply_cutomized_preprocess(org_img)
    data = {"image": preproc_img}
    processed = preproc(**data)
    
    inputs = np.zeros([1, *img_size])
    inputs[0] = np.expand_dims(processed["image"], axis=-1)
    output = model(inputs)
    output = (output[0].numpy()>0.5).astype(np.uint8).squeeze(-1)*255 
    
    return preproc_img, output  


def retrieve_AG_model_result(model, org_img, img_size):
                
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
            
    return  preproc_img, output

    
def retrieve_SG_model_result(model, org_img, img_size, resize_shape, prepoc):
    preproc_img = preprocess.apply_cutomized_preprocess(org_img)
    data = {"image": preproc_img}
    data = prepoc(**data)
    
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
    
def combine_image_with_output(img, output):
    # Resize the output to match the image size
    output_resized = cv2.resize(output, (img.shape[1], img.shape[0]))

    # Create a mask using the white regions of the resized output
    mask = (output_resized == 255).astype(np.uint8)

    # Create a copy of the original image
    combined = np.copy(img)

    # Set the white regions of the original image to white where the mask is 1
    combined[mask == 1] = [255, 255, 255]  # Assuming RGB image, modify as needed

    return combined

def combine_preproc_with_output(preproc, output):
    # Resize the output to match the image size
    output_resized = cv2.resize(output, (preproc.shape[1], preproc.shape[0]))
    
    # Create a mask using the white regions of the resized output
    mask = (output_resized == 255).astype(np.uint8)
    
    # Create a copy of the original image
    combined = np.copy(preproc)

    # Set the white regions of the original image to white where the mask is 1
    combined[mask == 1] = 1
 
    return combined


def combine_label_with_output(label, output):
    # # Resize the output to match the label size (assuming label and output have the same size)
    # output_resized = cv2.resize(output, (label.shape[1], label.shape[0]))
    
    # # Create a mask for white regions in the label
    # mask_label_white = (label == 255).astype(np.uint8)

    # # Create a mask for white regions in the output
    # mask_output_white = (output_resized == 255).astype(np.uint8)

    # # Create a mask for common white regions in both label and output
    # mask_common_white = np.minimum(mask_label_white, mask_output_white)
    
    # combined = np.zeros_like(label)
    
    # # Set white for common white regions in both label and output
    # combined[mask_common_white == 1] = 255

    # # Set green for white regions in label
    # combined[mask_label_white == 1] = 50  # Green (BGR)

    # # Set blue for white regions in output
    # combined[mask_output_white == 1] = 200  # Blue (BGR)
    # combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    
    # return combined_rgb
    
    output_resized = cv2.resize(output, (label.shape[1], label.shape[0]))
    
    return output_resized
    

def visualize_FG_result(model,
                        num_images_to_select, SOURCE,
                        img_size, preproc):
    input_path_list, label_path_list = retrieve_path_list(SOURCE, num_images_to_select)
    
    for i in range(0, num_images_to_select):
        file_name = os.path.basename(input_path_list[i])
        org_img = cv2.imread(input_path_list[i])
        label_img = cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE)
        preproc_img, output_img = retrieve_FG_model_result(model, org_img, img_size, preproc)
    
        plot_result_images_ver2(file_name, org_img, preproc_img ,label_img, output_img)
        
def visualize_AG_result(model,
                        num_images_to_select, SOURCE,
                        img_size):
    input_path_list, label_path_list = retrieve_path_list(SOURCE, num_images_to_select)
    
    for i in range(0, num_images_to_select):
        file_name = os.path.basename(input_path_list[i])
        org_img = cv2.imread(input_path_list[i])
        label_img = cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE)
        preproc_img, output_img = retrieve_AG_model_result(model, org_img, img_size)

        plot_result_images_ver2(file_name, org_img, preproc_img ,label_img, output_img)
        
def visualize_SG_result(model,
                        num_images_to_select, SOURCE,
                        img_size, resize_shape, preproc):
    
    input_path_list, label_path_list = retrieve_path_list(SOURCE, num_images_to_select)
    
    for i in range(0, num_images_to_select):
        file_name = os.path.basename(input_path_list[i])
        org_img = cv2.imread(input_path_list[i])
        label_img = cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE)
        preproc_img, output_img = retrieve_SG_model_result(model, org_img, img_size, resize_shape, preproc)
        
        plot_result_images_ver2(file_name, org_img, preproc_img ,label_img, output_img)
    
def plot_result_images_ver1(file_name, org, prepoc, label, result):
    # Plotting
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(org, cv2.COLOR_BGR2RGB))
    plt.title(file_name)
        
    #Preprocessed Image
    plt.subplot(1, 4, 2)
    plt.imshow(prepoc, cmap='gray')
    plt.title('Preprocessed Image')
        
    # Label Image
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))
    plt.title('Label Image')
        
    # Output Image
    plt.subplot(1, 4, 4)
    plt.imshow(result, cmap='gray')
    plt.title('Model Output')

    plt.show()
    
def plot_result_images_ver2(file_name, org, prepoc, label, result):
    # Plotting
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(org, cv2.COLOR_BGR2RGB))
    plt.title(file_name)
        
    #Preprocessed Image
    plt.subplot(2, 4, 2)
    plt.imshow(prepoc, cmap='gray')
    plt.title('Preprocessed Image')
        
    # Label Image
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))
    plt.title('Label Image')
        
    # Output Image
    plt.subplot(2, 4, 4)
    output_resized = cv2.resize(result, (label.shape[1], label.shape[0]))
    plt.imshow(output_resized, cmap='gray')
    plt.title('Model Output')
    
    # Combination Original Image
    plt.subplot(2, 4, 5)
    combined_org = combine_image_with_output(org, result)
    plt.imshow(cv2.cvtColor(combined_org, cv2.COLOR_BGR2RGB))
    plt.title(file_name)
        
    # Combination Preprocessed Image
    plt.subplot(2, 4, 6)
    combined_preproc = combine_preproc_with_output(prepoc, result)
    plt.imshow(combined_preproc, cmap='gray')
    plt.title('Combination Preprocessed Image')
        
    # Combination Label Image
    plt.subplot(2, 4, 7)
    combined_label = combine_label_with_output(label, result)
    plt.imshow(combined_label, cmap='gray')
    plt.title('Combination Label Image')

    plt.show()

def plot_history(history_path):
    with open(history_path, 'r') as json_file:
        history = json.load(json_file)

    # Assuming the structure of your history file is like {'accuracy': [...], 'val_accuracy': [...], 'loss': [...], 'val_loss': [...]}

    sensitivity = history['sensitivity']
    val_sensitivity = history['val_sensitivity']
    specificity = history['specificity']
    val_specificity = history['val_specificity']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']

    epochs = range(1, len(accuracy) + 1)                                  
                           
    plt.plot(epochs, accuracy, "bo", label="Train Accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validatin Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.figure()                            
                        
    plt.plot(epochs, sensitivity, "go", label="sensitivity")
    plt.plot(epochs, val_sensitivity, "g", label="Validatin sensitivity")
    plt.title("sensitivity")
    plt.legend()
    plt.figure()  
    
    plt.plot(epochs, specificity , "go", label="specificity")
    plt.plot(epochs, val_specificity , "g", label="Validatin specificity")
    plt.title("specificity")
    plt.legend()
    plt.figure()  