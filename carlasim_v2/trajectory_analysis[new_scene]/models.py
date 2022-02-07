import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, add, Input, GlobalAveragePooling1D, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

def conv_lstm(input_shape, target, nb_classes = None):
    seq_length = 30

    model = Sequential(
        [
            Conv1D(filters = 64, kernel_size = 5, padding = 'same', input_shape = (seq_length, 2)),
            BatchNormalization(),
            Activation(activation = 'relu'),
            MaxPooling1D(pool_size = 2),
            LSTM(seq_length),
        ]
    )
    if target == 'next_cam':
        model.add(Dense(nb_classes, activation = 'softmax'))
    else:
        model.add(Dense(1, activation = 'linear'))

    return model

# class ResNet(keras.models.Model):
#     def __init__(self, task, nb_classes, n_feature_maps = 64):
#         super().__init__()

#         #self.add = add()

#         ## Block 1
#         self.conv1 = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')
#         self.bn1 = BatchNormalization()
#         self.act = LeakyReLU()

#         self.conv2 = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')
#         self.bn2 = BatchNormalization()
        
#         self.conv3 = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')
#         self.bn3 = BatchNormalization()

#         ## shortcut 1
#         self.conv_shortcut1 = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')
#         self.bn_shortcut1 = BatchNormalization()

#         ## Block 2

#         self.conv4 = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')
#         self.bn4 = BatchNormalization()
        
#         self.conv5 = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')
#         self.bn5 = BatchNormalization()

#         self.conv6 = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')
#         self.bn6 = BatchNormalization()

#         ## shortcut 2

#         self.conv_shortcut2 = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')
#         self.bn_shortcut2 = BatchNormalization()

#         ## Block 3

#         self.conv7 = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')
#         self.bn7 = BatchNormalization()

#         self.conv8 = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')
#         self.bn8 = BatchNormalization()

#         self.conv9 = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')
#         self.bn9 = BatchNormalization()

#         ## Shortcut 3

#         self.bn_shortcut3 = BatchNormalization()

#         self.global_pool = GlobalAveragePooling1D()
#         if task == 'next_cam':
#             self.out = Dense(nb_classes, 'softmax')
#         elif task == 'time':
#             self.out = Dense(1, 'linear')
    
#     def call(self, x):

#         ## Block 1

#         block1 = self.conv1(x)
#         block1 = self.bn1(block1)
#         block1 = self.act(block1)

#         block1 = self.conv2(block1)
#         block1 = self.bn2(block1)
#         block1 = self.act(block1)

#         block1 = self.conv3(block1)
#         block1 = self.bn3(block1)

#         shortcut1 = self.conv_shortcut1(x)
#         shortcut1 = self.bn_shortcut1(shortcut1)
        
#         block1 = add([shortcut1, block1])
#         block1 = self.act(block1)

#         ## Block 2

#         block2 = self.conv4(block1)
#         block2 = self.bn4(block2)
#         block2 = self.act(block2)

#         block2 = self.conv5(block2)
#         block2 = self.bn5(block2)
#         block2 = self.act(block2)

#         block2 = self.conv6(block2)
#         block2 = self.bn6(block2)

#         shortcut2 = self.conv_shortcut2(block1)
#         shortcut2 = self.bn_shortcut2(shortcut2)

#         block2 = add([shortcut2, block2])
#         block2 = self.act(block2)

#         ## Block 3

#         block3 = self.conv7(block2)
#         block3 = self.bn7(block3)
#         block3 = self.act(block3)

#         block3 = self.conv8(block3)
#         block3 = self.bn8(block3)
#         block3 = self.act(block3)

#         block3 = self.conv8(block3)
#         block3 = self.bn8(block3)

#         shortcut3 = self.bn_shortcut3(block2)

#         block3 = add([block3, shortcut3])
#         block3 = self.act(block3)

#         pool = self.global_pool(block3)
#         o = self.out(pool)

#         return o

def ResNet(input_shape, target, nb_classes = 1, n_feature_maps=64):
    input_layer = Input(input_shape)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = BatchNormalization()(conv_x)
    conv_x = LeakyReLU()(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = LeakyReLU()(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = add([shortcut_y, conv_z])
    output_block_1 = LeakyReLU()(output_block_1)

    # BLOCK 2

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = LeakyReLU()(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = LeakyReLU()(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_2 = add([shortcut_y, conv_z])
    output_block_2 = LeakyReLU()(output_block_2)

    # BLOCK 3

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = LeakyReLU()(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = LeakyReLU()(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization()(output_block_2)

    output_block_3 = add([shortcut_y, conv_z])
    output_block_3 = LeakyReLU()(output_block_3)

    # FINAL

    gap_layer = GlobalAveragePooling1D()(output_block_3)

    if target == 'next_cam':
        output_layer = Dense(nb_classes, activation='softmax')(gap_layer)
    else:
        output_layer = Dense(1, activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
    #             metrics=['accuracy'])

    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    return model

# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true))) 

