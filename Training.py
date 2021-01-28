import tensorflow as tf
from tensorflow import keras as keras

### TPU initialization at program start
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

from ImageLoader import loadTFRecImages, loadJpgImages
from GANModel import *

BATCH_SIZE_PER_REPLICA = 30
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10
BUFFER_SIZE = 2048
IMAGE_SIZE = [256, 256]

GAN_MODEL = 'cyclic'

def loadData() -> (tf.data.Dataset, tf.data.Dataset):

    ### Load data and distribute
    photos = loadTFRecImages("photo_tfrec").shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    style = loadTFRecImages("monet_tfrec").shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

    # photos = loadJpgImages("photo_jpg").shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    # style = loadJpgImages("monet_jpg").shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

    return photos, style

def train(photos: tf.data.Dataset, style: tf.data.Dataset) -> keras.Model:

    ### Strategy scope
    with strategy.scope():

        ### Siamese GAN
        if GAN_MODEL == 'siamese':

            ## Losses
            euclDistProxLoss = euclideanDistanceImageProximalLoss
            euclDistDistLoss = euclideanDistanceImageDistalLoss
            cosSimPairLoss = cosineSimilarityImagePairLoss

            # Model
            optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, epsilon=0.1)
            model = SiameseGAN([*IMAGE_SIZE, 3], 0.2, euclDistProxLoss, euclDistDistLoss, cosSimPairLoss,
                               training = True)
            model.compile(optimizer)

        ### Cycle GAN
        elif GAN_MODEL == 'cyclic':

            ## Generator/discriminator
            genPhotoFn = Generator([*IMAGE_SIZE, 3], True)
            genStyleFn = Generator([*IMAGE_SIZE, 3], True)
            discPhotoFn = Discriminator([*IMAGE_SIZE, 3], True)
            discStyleFn = Discriminator([*IMAGE_SIZE, 3], True)

            ## Losses
            cycleLossFn = identityLoss()
            genLossFn = generatorLoss()
            discLossFn = discriminatorLoss()
            identLossFn = identityLoss()

            ## Optimizers
            genPhotoOptimFn = adamOptimizer()
            genStyleOptimFn = adamOptimizer()
            discPhotoOptimFn = adamOptimizer()
            discStyleOptimFn = adamOptimizer()

            ## Model
            model = CycleGAN([*IMAGE_SIZE, 3],
                genPhotoFn, genStyleFn, discPhotoFn, discStyleFn,
                cycleLossFn, genLossFn, discLossFn, identLossFn,
                genPhotoOptimFn, genStyleOptimFn, discPhotoOptimFn, discStyleOptimFn)
            model.compile()

    ## Train
    model.fit(tf.data.Dataset.zip((photos, style)), epochs=EPOCHS, verbose=1)
    return model

def generate(photos: tf.data.Dataset, model: keras.Model) -> tf.data.Dataset:
    model.predict(photos)

if __name__ == '__main__':
    photos, style = loadData()
    train(photos, style)




