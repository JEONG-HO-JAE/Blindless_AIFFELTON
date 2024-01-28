import sys, json
sys.path.append("/visuworks/Blindless_AIFFELTON/Models") 
import loss, metrics

import tensorflow as tf
import os
from keras.callbacks import Callback
from keras.optimizers.schedules import CosineDecay
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint

# Define custom objects for loading the model
custom_objects = {'DiceLoss': loss.DiceLoss(), 
                  'sensitivity': metrics.sensitivity,
                  'specificity': metrics.specificity,
                  'accuracy' : metrics.accuracy}
class SaveModelEveryEpoch(Callback):
    def __init__(self, save_path, first_epoch=1):
        super(SaveModelEveryEpoch, self).__init__()
        self.save_path = save_path # Path where you want to save the models
        self.first_epoch = first_epoch

    def on_epoch_end(self, epoch, logs=None):
        file_name = "{}--{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pb".format(self.first_epoch + epoch, logs['loss'], logs['sensitivity'],logs['specificity'],
                                                                logs['val_sensitivity'], logs['val_specificity'])
        model_save_path = os.path.join(self.save_path, file_name)

        self.model.save(model_save_path)
        print(f'Model saved to {model_save_path}')

    def get_config(self):
        config = super(SaveModelEveryEpoch, self).get_config()
        config.update({'save_path': self.save_path, 'first_epoch': self.first_epoch})
        return config
    
class WeightDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, weight_decay, verbose=0):
        super(WeightDecayScheduler, self).__init__()
        self.schedule = schedule
        self.weight_decay = weight_decay
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        decay = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr * (1.0 - self.weight_decay))
        if self.verbose > 0:
            print('\nEpoch %05d: WeightDecayScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr * (1.0 - self.weight_decay)))
            
def cosine_decay_with_warmup(epoch, lr):
    initial_learning_rate = 1e-4
    warmup_epochs = 5
    cosine_decay_epochs = 40
    warmup_lr = (initial_learning_rate / warmup_epochs) * epoch
    if epoch < warmup_epochs:
        return warmup_lr
    else:
        return 0.5 * initial_learning_rate * (1 + tf.math.cos((epoch - warmup_epochs) / (cosine_decay_epochs - warmup_epochs) * 3.141592653589793))
               
def model_train(model, epoch,
                train_generator, test_generator, 
                model_path, history_path,
                save_every_epoch, apply_weight_decay):
    
    weight_decay = 1e-5
    
    weight_decay_callback = WeightDecayScheduler(schedule=cosine_decay_with_warmup,
                                                 weight_decay=weight_decay,
                                                 verbose=1)
    
    save_model_callback = SaveModelEveryEpoch(model_path, epoch)
    
    callbacks_list = []
    if save_every_epoch:
        callbacks_list.append(save_model_callback)
    if apply_weight_decay:
        callbacks_list.append(weight_decay_callback)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=loss.DiceLoss(), 
              metrics=[metrics.sensitivity, metrics.specificity, metrics.accuracy])

    # # Define ModelCheckpoint callback
    # checkpoint = ModelCheckpoint(
    #     model_path,
    #     monitor='val_sensitivity',  # Monitor validation sensitivity
    #     save_best_only=True,     
    #     mode='max',  # Save the model when validation sensitivity is at its highest
    #     verbose=1
    # )
    
    # Train the model with the ModelCheckpoint callback
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        callbacks=callbacks_list  # Pass the callback to the fit method
    )
    # Save history to JSON file
    with open(history_path, 'w') as json_file:
        json.dump(history.history, json_file)
    
