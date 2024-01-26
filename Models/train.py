import sys, json
sys.path.append("/visuworks/Blindless_AIFFELTON/Models") 
import loss, metrics

from keras.optimizers import *
from keras.callbacks import ModelCheckpoint

# Define custom objects for loading the model
custom_objects = {'DiceLoss': loss.DiceLoss(), 
                  'sensitivity': metrics.sensitivity,
                  'specificity': metrics.specificity,
                  'accuracy' : metrics.accuracy}

def model_train(model, epoch,
                train_generator, test_generator, 
                model_path, history_path):
    
    model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=loss.DiceLoss(), 
              metrics=[metrics.sensitivity, metrics.specificity, metrics.accuracy])

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
        epochs=epoch,
        callbacks=[checkpoint]  # Pass the callback to the fit method
    )
    # Save history to JSON file
    with open(history_path, 'w') as json_file:
        json.dump(history.history, json_file)
    
