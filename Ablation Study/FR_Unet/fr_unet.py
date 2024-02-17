import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

class ConvBlock(tf.keras.Model):
    def __init__(self, out_c, dp=0):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.Sequential([
            Conv2D(out_c, kernel_size=3, padding='same', use_bias=False),
            BatchNormalization(),
            Dropout(dp),
            LeakyReLU(alpha=0.1),
            Conv2D(out_c, kernel_size=3, padding='same', use_bias=False),
            BatchNormalization(),
            Dropout(dp),
            LeakyReLU(alpha=0.1)
        ])

    def call(self, inputs):
        return self.conv(inputs)

class FeatureFuse(tf.keras.Model):
    def __init__(self, out_c):
        super(FeatureFuse, self).__init__()
        self.conv11 = Conv2D(out_c, kernel_size=1, padding='same', use_bias=False)
        self.conv33 = Conv2D(out_c, kernel_size=3, padding='same', use_bias=False)
        self.conv33_di = Conv2D(out_c, kernel_size=3, padding='same', use_bias=False, dilation_rate=2)
        self.norm = BatchNormalization()

    def call(self, inputs):
        x1 = self.conv11(inputs)
        x2 = self.conv33(inputs)
        x3 = self.conv33_di(inputs)
        out = self.norm(x1 + x2 + x3)
        return out

class UpBlock(tf.keras.Model):
    def __init__(self, out_c, dp=0):
        super(UpBlock, self).__init__()
        self.up = tf.keras.Sequential([
            Conv2DTranspose(out_c, kernel_size=2, strides=2, padding='valid', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.1)
        ])

    def call(self, inputs):
        return self.up(inputs)

class DownBlock(tf.keras.Model):
    def __init__(self, out_c, dp=0):
        super(DownBlock, self).__init__()
        self.down = tf.keras.Sequential([
            Conv2D(out_c, kernel_size=2, strides=2, padding='valid', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.1)
        ])

    def call(self, inputs):
        return self.down(inputs)

class Block(tf.keras.Model):
    def __init__(self, out_c, dp=0, is_up=False, is_down=False, fuse=False):
        super(Block, self).__init__()
        self.fuse = FeatureFuse(out_c) if fuse else Conv2D(out_c, kernel_size=1, strides=1)
        self.is_up = is_up
        self.is_down = is_down
        self.conv = ConvBlock(out_c, dp=dp)
        self.up = UpBlock(out_c, dp=dp) if is_up else None
        self.down = DownBlock(out_c, dp=dp) if is_down else None

    def call(self, x):
        if self.is_up or self.is_down:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up and self.is_down:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down
        elif self.is_up:
            x_up = self.up(x)
            return x, x_up
        elif self.is_down:
            x_down = self.down(x)
            return x, x_down
        else:
            return x

class FR_UNet(tf.keras.Model):
    def __init__(self, num_classes=1, feature_scale=2, dropout=0.2, fuse=True, out_ave=True):
        super(FR_UNet, self).__init__()
        self.out_ave = out_ave
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = Block(filters[0], dp=dropout, is_down=True, fuse=fuse)
        self.block1_2 = Block(filters[0], dp=dropout, is_down=True, fuse=fuse)
        self.block1_1 = Block(filters[0], dp=dropout, is_down=True, fuse=fuse)
        
        self.block10 = Block(filters[0], dp=dropout, is_down=True, fuse=fuse)
        self.block11 = Block(filters[0], dp=dropout, is_down=True, fuse=fuse)
        self.block12 = Block(filters[0], dp=dropout, is_down=False, fuse=fuse)
        self.block13 = Block(filters[0], dp=dropout, is_down=False, fuse=fuse)
        
        self.block2_2 = Block(filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = Block(filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        
        self.block20 = Block(filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = Block(filters[1], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = Block(filters[1], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        
        self.block3_1 = Block(filters[2], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        
        self.block30 = Block(filters[2], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = Block(filters[2], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        
        self.block40 = Block(filters[3], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = Conv2D(num_classes, kernel_size=1, padding='valid', use_bias=True)
        self.final2 = Conv2D(num_classes, kernel_size=1, padding='valid', use_bias=True)
        self.final3 = Conv2D(num_classes, kernel_size=1, padding='valid', use_bias=True)
        self.final4 = Conv2D(num_classes, kernel_size=1, padding='valid', use_bias=True)
        self.final5 = Conv2D(num_classes, kernel_size=1, padding='valid', use_bias=True)
        self.fuse_layer = Conv2D(num_classes, kernel_size=1, padding='valid', use_bias=True)

    def call(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(tf.concat([x1_2, x_up2_2], axis=-1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(tf.concat([x_down1_2, x2_2], axis=-1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(tf.concat([x1_1, x_up2_1], axis=-1))
        x20, x_up20, x_down20 = self.block20(tf.concat([x_down1_1, x2_1, x_up3_1], axis=-1))
        x30, x_up30 = self.block30(tf.concat([x_down2_1, x3_1], axis=-1))
        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(tf.concat([x10, x_up20], axis=-1))
        x21, x_up21 = self.block21(tf.concat([x_down10, x20, x_up30], axis=-1))
        _, x_up31 = self.block31(tf.concat([x_down20, x30, x_up40], axis=-1))
        x12 = self.block12(tf.concat([x11, x_up21], axis=-1))
        _, x_up22 = self.block22(tf.concat([x_down11, x21, x_up31], axis=-1))
        x13 = self.block13(tf.concat([x12, x_up22], axis=-1))
        
        if self.out_ave:
            output = (self.final1(x1_1) + self.final2(x10) +
                      self.final3(x11) + self.final4(x12) + self.final5(x13)) / 5
        else:
            output = self.final5(x13)
            
        return output