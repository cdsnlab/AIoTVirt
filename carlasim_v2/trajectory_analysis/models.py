import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, add, Input, GlobalAveragePooling1D, LeakyReLU, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def transformer(
    task,
    n_classes,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    if task == 'nextcam':
        outputs = Dense(n_classes, activation="softmax")(x)
    else:
        outputs = Dense(1, activation = 'linear')(x)
    return keras.Model(inputs, outputs)

def conv_lstm(input_shape, target, nb_classes = None):
    seq_length = 30

    #input_layer = keras.Input(shape = input_shape)

    model = Sequential(
        [
            Conv1D(filters = 64, kernel_size = 5, padding = 'same', input_shape = (seq_length, 2)),
            BatchNormalization(),
            Activation(activation = 'relu'),
            MaxPooling1D(pool_size = 2),
            LSTM(seq_length),
        ]
    )
    if target == 'nextcam':
        model.add(Dense(nb_classes, activation = 'softmax'))
    else:
        model.add(Dense(1, activation = 'linear'))

    
    return model

# def conv_lstm(input_shape, target, nb_classes = None):
#     seq_length = 30

#     input_layer = keras.Input(shape = input_shape)

#     x = Conv1D(filters = 64, kernel_size = 5, padding = 'same', input_shape = (seq_length, 2))(input_layer)
#     x = BatchNormalization()(x)
#     x = Activation(activation = 'relu')(x)
#     x = MaxPooling1D(pool_size = 2)(x)
#     x = LSTM(seq_length)(x)

#     if target == 'nextcam':
#         outputs = Dense(nb_classes, activation = 'softmax')(x)
#     else:
#         outputs = Dense(1, activation = 'linear')(x)


#     return keras.Model(input_layer, outputs)


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

    if target == 'nextcam':
        output_layer = Dense(nb_classes, activation='softmax')(gap_layer)
    else:
        output_layer = Dense(1, activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
    #             metrics=['accuracy'])

    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    return model
