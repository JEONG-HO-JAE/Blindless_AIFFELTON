import sys, json
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
sys.path.append("/visuworks/Blindless_AIFFELTON/Models") 
import preprocess_full_size
import metrics
import loss
import data_generator_full_size
sys.path.append("/visuworks/Blindless_AIFFELTON/Models/Pyramid") 
import model

input_shape = (256, 256, 1) # Adjust based on your input data
model = model.unet_model(input_shape)

augmentation = preprocess_full_size.build_augmentation()
test_preproc = preprocess_full_size.build_augmentation(is_train=False)

train_generator = data_generator_full_size.FullSizedDataGenerator(
    data_generator_full_size.SOURCE, 
    augmentation=augmentation,
)
test_generator = data_generator_full_size.FullSizedDataGenerator(
    data_generator_full_size.SOURCE, 
    augmentation=test_preproc,
    is_train=False
)

model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=loss.DiceLoss(), 
              metrics=[metrics.sensitivity, metrics.specificity, metrics.accuracy])


model_path = '/visuworks/Blindless_AIFFELTON/script/Pyramid/model_parameters/50epochs_g_clahe.h5'
history_path = '/visuworks/Blindless_AIFFELTON/script/Pyramid/history/50epochs_g_clahe_history.json'

# Define ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_sensitivity',  # Monitor validation sensitivity
    save_best_only=True,     
    mode='max',  # Save the model when validation sensitivity is at its highest
    verbose=1
)
    
# Train the model with the ModelCheckpoint callback
history = model.fit(
    train_generator,
    validation_data=test_generator,
    steps_per_epoch=len(train_generator),
    epochs=80,
    callbacks=[checkpoint]  # Pass the callback to the fit method
)

# # Save history to JSON file
with open(history_path, 'w') as json_file:
    json.dump(history.history, json_file)