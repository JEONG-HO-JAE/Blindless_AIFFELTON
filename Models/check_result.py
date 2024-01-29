import cv2
import json, os, random
import preprocess
import numpy as np
import matplotlib.pyplot as plt

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
    # Initialize a new image with zeros
    
    output = cv2.resize(output, (label.shape[1], label.shape[0]))
    result_img = np.zeros_like(label)

    # Condition 1: label and output are both non-zero
    condition1 = (label != 0) & (output != 0)
    result_img[condition1] = 255

    # Condition 2: label is zero, output is non-zero
    condition2 = (label == 0) & (output != 0)
    result_img[condition2] = 100

    # Condition 3: label is non-zero, output is zero
    condition3 = (label != 0) & (output == 0)
    result_img[condition3] = 200

    # Create a blank RGB image
    colored_image = np.zeros((result_img.shape[0], result_img.shape[1], 3), dtype=np.uint8)

    # Set colors based on pixel values
    colored_image[result_img == 0] = [0, 0, 0]       # Black
    colored_image[result_img == 100] = [255, 0, 0]    # Red
    colored_image[result_img == 200] = [0, 255, 0]    # Green
    colored_image[result_img == 255] = [255, 255, 255] # White

    return colored_image

    
    

def visualize_FG_result(model,
                        num_images_to_select, SOURCE,
                        img_size, preproc):
    data = FG_result(model, num_images_to_select, SOURCE, img_size, preproc)
    for file_name, org_img, preproc_img ,label_img, output_img in data:
        plot_result_images_ver2(file_name, org_img, preproc_img ,label_img, output_img)
        
def visualize_AG_result(model,
                        num_images_to_select, SOURCE,
                        img_size):
    data = AG_result(model, num_images_to_select, SOURCE, img_size)
    for file_name, org_img, preproc_img ,label_img, output_img in data:
        plot_result_images_ver2(file_name, org_img, preproc_img ,label_img, output_img)
        
def visualize_SG_result(model,
                        num_images_to_select, SOURCE,
                        img_size, resize_shape, preproc):
    
    data = SG_result(model, num_images_to_select, SOURCE, img_size, resize_shape, preproc)
    for file_name, org_img, preproc_img ,label_img, output_img in data:
        plot_result_images_ver2(file_name, org_img, preproc_img ,label_img, output_img)


def FG_result(model, num_images_to_select, SOURCE, img_size, preproc):
    input_path_list, label_path_list = retrieve_path_list(SOURCE, num_images_to_select)
    files = []
    org_imgs = []
    label_imgs = []
    preproc_imgs = []
    output_imgs = []
    
    for i in range(0, num_images_to_select):
        files.append(os.path.basename(input_path_list[i]))
        org_img = cv2.imread(input_path_list[i])
        org_imgs.append(org_img)
        label_imgs.append(cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE))
        preproc_img, output_img = retrieve_FG_model_result(model, org_img, img_size, preproc)
        preproc_imgs.append(preproc_img)
        output_imgs.append(output_img)
        
    data = [_ for _ in zip(files, org_imgs, preproc_imgs, label_imgs, output_imgs)]
    
    return data

def AG_result(model, num_images_to_select, SOURCE, img_size):
    input_path_list, label_path_list = retrieve_path_list(SOURCE, num_images_to_select)
    files = []
    org_imgs = []
    label_imgs = []
    preproc_imgs = []
    output_imgs = []
    
    for i in range(0, num_images_to_select):
        files.append(os.path.basename(input_path_list[i]))
        org_img = cv2.imread(input_path_list[i])
        org_imgs.append(org_img)
        label_imgs.append(cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE))
        preproc_img, output_img = retrieve_AG_model_result(model, org_img, img_size)
        preproc_imgs.append(preproc_img)
        output_imgs.append(output_img)
        
    data = [_ for _ in zip(files, org_imgs, preproc_imgs, label_imgs, output_imgs)]
    
    return data
    
def SG_result(model, num_images_to_select, SOURCE, img_size, resize_shape, preproc):
    input_path_list, label_path_list = retrieve_path_list(SOURCE, num_images_to_select)
    files = []
    org_imgs = []
    label_imgs = []
    preproc_imgs = []
    output_imgs = []
    
    for i in range(0, num_images_to_select):
        files.append(os.path.basename(input_path_list[i]))
        org_img = cv2.imread(input_path_list[i])
        org_imgs.append(org_img)
        label_imgs.append(cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE))
        preproc_img, output_img = retrieve_SG_model_result(model, org_img, img_size, resize_shape, preproc)
        preproc_imgs.append(preproc_img)
        output_imgs.append(output_img)
        
    data = [_ for _ in zip(files, org_imgs, preproc_imgs, label_imgs, output_imgs)]
    
    return data
    
def compare_model_result(data_1, data_2, num_images_to_select):
    # Sort data based on filenames
    data_1_sorted = sorted(data_1, key=lambda x: x[0])
    data_2_sorted = sorted(data_2, key=lambda x: x[0])
    plot_models_result(data_1_sorted, data_2_sorted, num_images_to_select)
    

def plot_models_result(data_1_sorted, data_2_sorted, num_images_to_select):
    for i in range(0, num_images_to_select):
        file_name = data_1_sorted[i][0]
        org_img = data_1_sorted[i][1]
        preproc_img = cv2.resize(data_1_sorted[i][3], (org_img.shape[1], org_img.shape[0]))
        label_img = cv2.resize(data_1_sorted[i][2], (org_img.shape[1], org_img.shape[0]))
        
        output_img_1 = data_1_sorted[i][4]
        output_img_2 = data_1_sorted[i][4]
        
        label_img_1 = data_1_sorted[i][2]
        comb_img_1 = combine_image_with_output(org_img, output_img_1)
        label_img_2 = data_2_sorted[i][2]
        comb_img_2 = combine_image_with_output(org_img, output_img_2)
        
        pre_proc_1 = data_1_sorted[i][3]
        comb_pre_1 = combine_preproc_with_output(pre_proc_1, output_img_1)
        pre_proc_2 = data_2_sorted[i][3]
        comb_pre_2 = combine_preproc_with_output(pre_proc_2, output_img_2)
        
        comb_label_1 = combine_label_with_output(label_img_1, output_img_1)
        comb_label_2 = combine_label_with_output(label_img_2, output_img_2)

        
        plt.figure(figsize=(12, 4))
        
        # Original Image
        plt.subplot(3, 4, 1)
        plt.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
        plt.title(file_name)
        
        #Preprocessed Image
        plt.subplot(1, 4, 2)
        plt.imshow(preproc_img, cmap='gray')
        plt.title('Preprocessed Image')
        
        # Label Image
        plt.subplot(2, 4, 3)
        plt.imshow(cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB))
        plt.title('Label Image')
        
        # Combination Original Image
        plt.subplot(3, 4, 5)
        plt.imshow(cv2.cvtColor(comb_img_1, cv2.COLOR_BGR2RGB))
        plt.title(file_name)
            
        # Combination Preprocessed Image
        plt.subplot(3, 4, 6)
        plt.imshow(comb_pre_1, cmap='gray')
        plt.title('Combination Preprocessed Image')
            
        # Combination Label Image
        plt.subplot(3, 4, 7)
        plt.imshow(comb_label_1, cmap='gray')
        plt.title('Combination Label Image')
        
        plt.subplot(3, 4, 8)
        output_1_resized = cv2.resize(output_img_1, (label_img.shape[1], label_img.shape[0]))
        plt.imshow(output_1_resized, cmap='gray')
        plt.title('Model Output')
        
        # Combination Original Image
        plt.subplot(3, 4, 9)
        plt.imshow(cv2.cvtColor(comb_img_2, cv2.COLOR_BGR2RGB))
        plt.title(file_name)
            
        # Combination Preprocessed Image
        plt.subplot(3, 4, 10)
        plt.imshow(comb_pre_2, cmap='gray')
        plt.title('Combination Preprocessed Image')
            
        # Combination Label Image
        plt.subplot(3, 4, 11)
        plt.imshow(comb_label_2, cmap='gray')
        plt.title('Combination Label Image')
        
        plt.subplot(3, 4, 12)
        output_2_resized = cv2.resize(output_img_2, (label_img.shape[1], label_img.shape[0]))
        plt.imshow(output_2_resized, cmap='gray')
        plt.title('Model Output')

        plt.show()
        
        
    
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
    plt.imshow(label,cmap='gray')
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