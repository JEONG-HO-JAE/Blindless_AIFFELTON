import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, not_sigmoid=False, name='dice_loss', **kwargs):
        super(DiceLoss, self).__init__(name=name, **kwargs)
        self.not_sigmoid = not_sigmoid

    def dice_coefficient(self, y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
        dice_coeff = numerator / (denominator + tf.keras.backend.epsilon())  # Adding epsilon for numerical stability
        return dice_coeff

    def call(self, y_true, y_pred):
        # Calculate Dice Coefficient
        dice_coeff = self.dice_coefficient(y_true, y_pred)

        # Calculate Dice Loss
        dice_loss = 1.0 - dice_coeff

        return dice_loss