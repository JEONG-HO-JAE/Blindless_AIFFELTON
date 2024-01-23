import cv2
import json, os
import preprocess_full_size
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def visualize_result(model, preproc, image_path, label_path):
    org_img = cv2.imread(image_path)
    label_img = cv2.imread(label_path)

    img_size=(256, 256, 1)
    inputs = np.zeros([1, *img_size])
    
    data = {"image": org_img}
    processed = preproc(**data)
    preprocess_full_size.apply_preprocessing(processed)
    inputs[0] = processed["image"]
    output = model(inputs)
    output = (output[0].numpy()>0.5).astype(np.uint8).squeeze(-1)*255 
    output_img = Image.fromarray(output)
    
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
    plt.imshow(output_img, cmap='gray')
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