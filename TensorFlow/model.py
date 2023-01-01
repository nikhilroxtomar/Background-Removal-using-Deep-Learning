from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def residual_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)

    s = Conv2D(num_filters, 1, padding="same")(inputs)
    s = BatchNormalization()(s)
    x = Activation("relu")(x+s)

    return x

def dilated_conv(inputs, num_filters):
    x1 = Conv2D(num_filters, 3, padding="same", dilation_rate=3)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x2 = Conv2D(num_filters, 3, padding="same", dilation_rate=6)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x3 = Conv2D(num_filters, 3, padding="same", dilation_rate=9)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    x = Concatenate()([x1, x2, x3])
    x = Conv2D(num_filters, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters)
    return x

def build_model(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output
    s5 = resnet50.get_layer("conv4_block6_out").output
    # print(s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)

    """ Bridge """
    b1 = dilated_conv(s5, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    # print(d1.shape, d2.shape, d3.shape, d4.shape)

    y1 = UpSampling2D((8, 8), interpolation="bilinear")(d1)
    y1 = Conv2D(1, 1, padding="same", activation="sigmoid")(y1)

    y2 = UpSampling2D((4, 4), interpolation="bilinear")(d2)
    y2 = Conv2D(1, 1, padding="same", activation="sigmoid")(y2)

    y3 = UpSampling2D((2, 2), interpolation="bilinear")(d3)
    y3 = Conv2D(1, 1, padding="same", activation="sigmoid")(y3)

    y4 = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    outputs = Concatenate()([y1, y2, y3, y4])

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_model(input_shape)
    model.summary()








##
