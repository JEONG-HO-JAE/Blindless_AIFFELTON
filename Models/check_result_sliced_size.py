import cv2
import json, os
import preprocess_full_size
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def visualize_result(model, preproc, image_path, label_path):
    org_img = cv2.imread(image_path)
    label_img = cv2.imread(label_path)

    input_size=(64, 64, 1)
    full_img_size=(256, 256, 1)
    
    inputs = np.zeros([16, *input_size])
    outputs = np.zeros([16, *input_size])
    output = np.zeros(full_img_size, dtype=np.uint8)
    
    data = {"image": org_img}
    processed = preproc(**data)
    preprocess_full_size.apply_preprocessing(processed)
    
    for i in range(0, 16):
        start_row, end_row = 64 * (i // 4), 64 * (i // 4 + 1) 
        start_col, end_col = 64 * (i % 4), 64 * (i % 4 + 1)
        inputs[i] = processed["image"][start_row:end_row, start_col:end_col]
        
    outputs= model(inputs)
    
    for i in range(0, 16):
        start_row, end_row = 64 * (i // 4), 64 * (i // 4 + 1) 
        start_col, end_col = 64 * (i % 4), 64 * (i % 4 + 1)
        # print(start_row, end_col, start_col, end_col)
        outputs_np = (outputs[i].numpy()>0.5).astype(np.uint8)*255
        # print(outputs_np.shape)
        output[start_row:end_row, start_col:end_col] = outputs_np

    # Plotting
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
    plt.title(os.path.basename(image_path))
    
    plt.subplot(1, 4, 2)
     # Apply gamma correction
    data["image"] = preprocess_full_size.apply_gamma_correction(data["image"], mid=1.0)
    # Apply CLAHE (replace with your implementation)
    data["image"] = preprocess_full_size.apply_clahe(data["image"],
                                           preprocess_full_size.CLIPLIMIT, 
                                           preprocess_full_size.GRIDSIZE)
    plt.imshow(Image.fromarray(np.squeeze(data["image"]).astype('uint8')), cmap='gray')
    plt.title('Preprocessed Image')
    
    # Label Image
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB))
    plt.title('Label Image')
    
    # Output Image
    plt.subplot(1, 4, 4)
    plt.imshow(output, cmap='gray')
    plt.title('Model Output')

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