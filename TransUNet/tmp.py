import tensorflow as tf
# from tfwrapper import layers
import models.encoder_layers as encoder_layers
import models.decoder_layers as decoder_layers
from models.resnet_v2 import ResNetV2
import math

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math
tfkc = tfk.callbacks


def TransUNet(images, training, nlabels):
    trainable = True
    try:
         trainable = training.item()
    except:
         trainable = True

    # Tranformer Encoder
    x = images
    # Embedding
    # print(training.eval())
    # if trainable:
    resnet50v2 = ResNetV2(
        block_units=(3,4,9))
    y, features = resnet50v2(x)
    # else:
    #     resnet50v2, features = resnet_embeddings(x)
    #     y = resnet50v2.get_layer("conv4_block6_preact_relu").output
    #     x = resnet50v2.input

    y = tfkl.Conv2D(
        filters=768,
        kernel_size=1,
        strides=1,
        padding="valid",
        name="embedding",
        trainable=trainable
    )(y)
    y = tfkl.Reshape(
        (y.shape[1] * y.shape[2], 768))(y)
    y = encoder_layers.AddPositionEmbs(
        name="Transformer/posembed_input", trainable=trainable)(y)

    y = tfkl.Dropout(0.1)(y)

    # Transformer/Encoder
    for n in range(12):
        y, _ = encoder_layers.TransformerBlock(
            n_heads=12,
            mlp_dim=3072,
            dropout=0.1,
            name=f"Transformer/encoderblock_{n}",
            trainable=trainable,
        )(y, training = training)
    y = tfkl.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)

    n_patch_sqrt = int(math.sqrt(int(y.shape[1])))

    y = tfkl.Reshape(
        target_shape=[n_patch_sqrt, n_patch_sqrt, 768])(y)

    # Decoder CUP
    # if "decoder_channels" in self.config:
    y = decoder_layers.DecoderCup(
        decoder_channels=[256, 128, 64, 16], n_skip=3)(y, features)

    # Segmentation Head
    y = decoder_layers.SegmentationHead(
        filters=nlabels, kernel_size=1, upsampling_factor=1)(y)

    return y

if __name__ == '__main__':
    tf.config.list_physical_devices('GPU')