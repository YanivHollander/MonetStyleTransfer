import tensorflow as tf
from tensorflow.keras import layers as layers
from tensorflow import keras as keras
from tensorflow import Tensor
from typing import List, Callable
import unittest
import numpy as np
import tensorflow_addons as tfa

STRIDES = 2
KERNEL_SIZE = 4

### Models
class SmallGAN(keras.Model):
    def train_step(self, data):
        return {'loss': 0}

    def __call__(self, *args, **kwargs):
        photos = args[0]
        photo = next(photos.as_numpy_iterator())
        return photo

## Model parts
def endcodingLayer(filters: int, size: int, strides: int, intanceNorm = True) -> keras.Sequential:
    ret = keras.Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)
    ret.add(layers.Conv2D(filters=filters, kernel_size=size, strides=strides, padding='same',
                          kernel_initializer=initializer, use_bias=False))
    if intanceNorm:
        gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        ret.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    ret.add(layers.LeakyReLU())
    return ret

def decodingLayer(filters: int, size: int, strides: int, dropout = False) -> keras.Sequential:
    ret = keras.Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)
    ret.add(layers.Conv2DTranspose(filters=filters, kernel_size=size, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    ret.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    if dropout:
        ret.add(layers.Dropout(0.5))
    ret.add(layers.ReLU())
    return ret

def Generator(shape: List[int], training = False) -> keras.Model:
    print("training = ", training)

    inputs=layers.Input(shape=shape)
    x = inputs

    # Encoding stack of layers
    endcodingStack = []
    endcodingStack.append(endcodingLayer(64 , KERNEL_SIZE, STRIDES, intanceNorm=False))
    endcodingStack.append(endcodingLayer(128, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(256, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(512, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(512, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(512, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(512, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(512, KERNEL_SIZE, STRIDES))
    skipConnections = []
    for layer in endcodingStack:
        x = layer(x)
        skipConnections.append(x)
    skipConnections.pop()

    # Decoding stack of layers
    decodingStack = []
    decodingStack.append(decodingLayer(512, KERNEL_SIZE, STRIDES, dropout = training))
    decodingStack.append(decodingLayer(512, KERNEL_SIZE, STRIDES, dropout = training))
    decodingStack.append(decodingLayer(512, KERNEL_SIZE, STRIDES, dropout = training))
    decodingStack.append(decodingLayer(512, KERNEL_SIZE, STRIDES))
    decodingStack.append(decodingLayer(256, KERNEL_SIZE, STRIDES))
    decodingStack.append(decodingLayer(128 , KERNEL_SIZE, STRIDES))
    decodingStack.append(decodingLayer(64 , KERNEL_SIZE, STRIDES))
    for layer in decodingStack:
        x = layer(x)
        x = layers.Concatenate()([x, skipConnections.pop()])

    # Output layer - tanh activation to guarantee pixel range of [-1, 1]
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(filters=shape[2], kernel_size=KERNEL_SIZE, strides=STRIDES, padding='same',
                               kernel_initializer=initializer, activation='tanh')(x)
    return keras.Model(inputs=inputs, outputs=x, name="generator")

def Discriminator(shape: List[int]) -> keras.Model:
    inputs = layers.Input(shape=shape)

    # Encoding stack of layers
    endcodingStack = []
    endcodingStack.append(endcodingLayer(64 , KERNEL_SIZE, STRIDES, intanceNorm = False))
    endcodingStack.append(endcodingLayer(128, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(256, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(256, KERNEL_SIZE, STRIDES))
    endcodingStack.append(endcodingLayer(1, KERNEL_SIZE, STRIDES))

    # A input
    x = inputs
    for i, layer in enumerate(endcodingStack):
        x = layer(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(16, name='dense')(x) # Last layer outputs logits    # FIXME: Determine optimal size of vector
    x = tfa.layers.InstanceNormalization(center=False, scale=False)(x)

    return keras.Model(inputs=inputs, outputs=x, name='discriminator')

### Losses that go along with SiameseGAN model
def euclideanImageProximalLoss() -> Callable[[Tensor], Tensor]:
    """
    Calculates the average proximal (from origin) Euclidean loss along batch of vectors
    :param x: A bach of vectors: (batch, size)
    :return:
    """
    return lambda x: tf.norm(x, axis=1)

def euclideanImageDistalLoss(margin) -> Callable[[Tensor], Tensor]:
    """
    Calculates the average distal (around a margin) Euclidean loss along batch of vectors
    :param x:       A bach of vectors: (batch, size)
    :param margin:  Radial margin
    :return:
    """
    return lambda x: tf.math.maximum(0., margin - tf.norm(x, axis=1))

def cosineSimilarityImagePairLoss() -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    """
    Calculates cosine similarity between pairs of original and generated images, after run by discriminator
    :param x1:  1st original image, forward modeled by discriminator
    :param x2:  2nd original image, forward modeled by discriminator
    :param y1:  1st image generated from original, forward modeled by discriminator
    :param y2:  2nd image generated from original, forward modeled by discriminator
    :return: COSINE SIMILARITY LOSS
    """
    return lambda x1, x2, y1, y2: \
        tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)(x1-x2, y1-y2)

### Siamese GAN model
class SiameseGAN(keras.Model):
    def __init__(self,
                 genStyle: keras.Model,
                 discStyle: keras.Model,
                 euclProxLoss: Callable[[Tensor], Tensor],
                 euclDistLoss: Callable[[Tensor], Tensor],
                 cosSimPairLoss: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
                 genStyleOptim: tf.keras.optimizers.Adam,
                 discStyleOptim: tf.keras.optimizers.Adam):
        super(SiameseGAN, self).__init__()

        # Model parts
        self.genStyle = genStyle
        self.discStyle = discStyle

        # Losses
        self.euclProxLoss = euclProxLoss
        self.euclDistLoss = euclDistLoss
        self.cosSimPairLoss = cosSimPairLoss

        # Optimizers
        self.genStyleOptim = genStyleOptim
        self.discStyleOptim = discStyleOptim

    def train_step(self, data):
        photos, photos_s, styles = data

        with tf.GradientTape(persistent=True) as tape:
            fakeStyle = self.genStyle(photos, training=True)        # Photos -> fake styles
            fakeStyle_s = self.genStyle(photos_s, training=True)    # Shuffled photos -> fake shuffled styles
            discFakeStyle = self.discStyle(fakeStyle)               # Discriminator of fake styles
            discFakeStyle_s = self.discStyle(fakeStyle_s)           # Discriminator of fake shuffled styles
            discStyle = self.discStyle(styles)                      # Discriminator of styles
            discPhoto = self.discStyle(photos)                      # Discriminator of photos
            discPhoto_s = self.discStyle(photos_s)                  # Discriminator of shuffled photos

            # Discriminator losses
            lossD1 = self.euclProxLoss(discStyle)
            lossD2 = self.euclDistLoss(discFakeStyle)
            lossD3 = self.cosSimPairLoss(discPhoto, discPhoto_s, discFakeStyle, discFakeStyle_s)
            lossD = lossD1 + lossD2 + lossD3

            # Generator losses
            lossG1 = self.euclProxLoss(discFakeStyle)
            lossG = lossG1 + lossD3
        DiscriminatorGradients = tape.gradient(lossD, self.discStyle.trainable_variables)    # Discriminator gradient
        GeneratorGradients = tape.gradient(lossG, self.genStyle.trainable_variables)         # Generator gradient

        # Optimizer
        self.discStyleOptim.apply_gradients(zip(DiscriminatorGradients, self.discStyle.trainable_variables))
        self.genStyleOptim.apply_gradients(zip(GeneratorGradients, self.genStyle.trainable_variables))

        return {"lossD": lossD, "lossD1": lossD1, "lossD2": lossD2, "lossD3": lossD3, "lossG": lossG, "lossG1": lossG1}

### Losses that go along with cycle GAN model
def flatten(x: Tensor) -> Tensor:
    return tf.reshape(x, [tf.shape(x)[0], 1, -1])

def generatorLoss() -> Callable[[Tensor], Tensor]:
    return lambda x: \
                        tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) \
                        (tf.ones_like(flatten(x)), flatten(x))

def discriminatorLoss() -> Callable[[Tensor, Tensor], Tensor]:
    origLoss = lambda x: \
                        tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) \
                        (tf.ones_like(flatten(x)), flatten(x))
    fakeLoss = lambda x: \
                        tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) \
                        (tf.zeros_like(flatten(x)), flatten(x))
    return lambda orig, fake: 0.5 * (origLoss(orig) + fakeLoss(fake))

def identityLoss(lmbd: float) -> Callable[[Tensor, Tensor], Tensor]:
    return lambda x, y: \
            lmbd * tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE) \
            (flatten(x), flatten(y))

def adamOptimizer() -> tf.keras.optimizers.Adam:
    return tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

### Cycle GAN
class CycleGAN(keras.Model):
    def __init__(self,
                 genPhoto: keras.Model,
                 genStyle: keras.Model,
                 discPhoto: keras.Model,
                 discStyle: keras.Model,
                 cycleLoss: tf.keras.losses,
                 genLoss: Callable[[Tensor], Tensor],
                 discLoss: Callable[[Tensor, Tensor], Tensor],
                 identLoss: tf.keras.losses,
                 genPhotoOptim: tf.keras.optimizers.Adam,
                 genStyleOptim: tf.keras.optimizers.Adam,
                 discPhotoOptim: tf.keras.optimizers.Adam,
                 discStyleOptim: tf.keras.optimizers.Adam):
        super(CycleGAN, self).__init__()

        # Model parts
        self.genPhoto = genPhoto
        self.genStyle = genStyle
        self.discPhoto = discPhoto
        self.discStyle = discStyle

        # Losses
        self.cycleLoss = cycleLoss
        self.genLoss = genLoss
        self.discLoss = discLoss
        self.identLoss = identLoss

        # Optimizers
        self.genPhotoOptim = genPhotoOptim
        self.genStyleOptim = genStyleOptim
        self.discPhotoOptim = discPhotoOptim
        self.discStyleOptim = discStyleOptim

    def train_step(self, data):
        photos, styles = data

        with tf.GradientTape(persistent=True) as tape:

            ## Cycle between photo and style domains
            # Generating stylish photos with style generator, and going back with photo generator
            fakeStyle = self.genStyle(photos, training=True)         # Photo -> fake styles
            cyclePhoto = self.genPhoto(fakeStyle, training=True)     # Fake styles -> photos

            # Creating a photo from style, and going back to style domain
            fakePhoto =  self.genPhoto(styles, training=True)        # Style -> fake photo
            cycleStyle = self.genStyle(fakePhoto, training=True)     # Fake photo -> style

            # Cycle consistency loss: mean absolute difference loss between photo/style and its cycled version ()
            cycleLoss = self.cycleLoss(photos, cyclePhoto) + self.cycleLoss(styles, cycleStyle)

            ## Discriminator fake style and photo
            discFakeStyle = self.discStyle(fakeStyle)
            genStyleLoss0 = self.genLoss(discFakeStyle)     # Penalizing for identifying a fake style
            discFakePhoto = self.discPhoto(fakePhoto)
            genPhotoLoss0 = self.genLoss(discFakePhoto)     # Penalizing for identifying a fake photo

            # Creating styles from styles, and photos from photos
            sameStyle = self.genStyle(styles)          # Style -> fake style
            samePhoto = self.genPhoto(photos)          # Photo -> fake photo

            # Penalizing generators for:
            # 1) Fake style/photo. Create fakes that can be identified by the discriminator;
            # 2) Being inconsistent. Style and photo generators should be the inverse of each other;
            # 3) Change distribution of self. Style input to style generator should remain style - same for photos
            genStyleLoss = genStyleLoss0 + cycleLoss + self.identLoss(styles, sameStyle)
            genPhotoLoss = genPhotoLoss0 + cycleLoss + self.identLoss(photos, samePhoto)

            # Penalizing discriminators for:
            # 1) Fail to discriminate between real and fake style images
            # 2) Fail to discriminate between real and fake photo images
            discStyle = self.discStyle(styles)
            discStyleLoss = self.discLoss(discStyle, discFakeStyle)
            discPhoto = self.discPhoto(photos)
            discPhotoLoss = self.discLoss(discPhoto, discFakePhoto)

        ## Gradients
        genStyleGradients = tape.gradient(genStyleLoss, self.genStyle.trainable_variables)
        genPhotoGradients = tape.gradient(genPhotoLoss, self.genPhoto.trainable_variables)
        discStyleGradients = tape.gradient(discStyleLoss, self.discStyle.trainable_variables)
        discPhotoGradients = tape.gradient(discPhotoLoss, self.discPhoto.trainable_variables)

        ## Optimizer
        self.genStyleOptim.apply_gradients(zip(genStyleGradients, self.genStyle.trainable_variables))
        self.genPhotoOptim.apply_gradients(zip(genPhotoGradients, self.genPhoto.trainable_variables))
        self.discStyleOptim.apply_gradients(zip(discStyleGradients, self.discStyle.trainable_variables))
        self.discPhotoOptim.apply_gradients(zip(discPhotoGradients, self.discPhoto.trainable_variables))

        return({"genStyleLoss": genStyleLoss, "discStyleLoss": discStyleLoss, "genPhotoLoss": genPhotoLoss,
                "discPhotoLoss": discPhotoLoss})

### Tests
class TestGANModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_generator(self):
        generator=Generator([256, 256, 3])
        generator.summary()

    def test_siamese_discriminator(self):
        siamese_discriminator=Discriminator([256, 256, 3])
        siamese_discriminator.summary()

    def test_euclidean_loss(self):
        vect = np.float32(np.random.rand(4, 3))     # A batch of 4 vectors size 3
        eucLoss = euclideanDistanceFromOrigin(vect)
        print(vect)
        print(eucLoss)
        print(tf.nn.compute_average_loss(eucLoss))

    def test_cosine_similarity_loss(self):
        vect1 = np.float32(np.random.rand(4, 3))
        vect2 = np.float32(np.random.rand(4, 3))
        print("vect1 = ", vect1)
        print("vect2 = ", vect2)
        print("Cosine similarity: ", cosineSimilarityImagePairLoss(vect1, vect2))

        y_true = np.array([[0., 1.], [1., 1.]])
        y_pred = np.array([[1., 0.], [1., 1.]])
        print(cosineSimilarityImagePairLoss(y_true, y_pred).numpy())

    def test_train_step(self):
        GANModel = SiameseGAN([256, 256, 3], 0.2)
        GANModel.compile()
        orig = np.float32(np.random.rand(10, 256, 256, 3))
        style = np.float32(np.random.rand(10, 256, 256, 3))
        data = (orig, style)
        GANModel.train_step(data)

    def test_cycleGAN_Losses(self):

        # Generator loss
        print("Generator loss")
        genLoss = generatorLoss()
        vects = np.float32(np.random.rand(5, 4, 2, 3))
        # print(genLoss(vects).numpy())
        ones = np.float32(np.ones_like(vects))
        print(genLoss(ones).numpy())

        # Discriminator loss
        print("Discriminator loss")
        zeros = np.zeros_like(vects)
        discLoss = discriminatorLoss()
        print(discLoss(zeros, zeros).numpy())
        # print(discLoss(zeros, ones).numpy())
        # print(discLoss(ones, zeros).numpy())
        # print(discLoss(ones, ones).numpy())

        # Identity loss
        print("Identity loss")
        print(identityLoss()(ones, zeros).numpy())

if __name__ == '__main__':
    unittest.main()
