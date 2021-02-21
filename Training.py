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

from ImageLoader import loadTFRecImages
from GANModel import *

BATCH_SIZE_PER_REPLICA = 1
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 1
BUFFER_SIZE = 2048
IMAGE_SIZE = [256, 256]

def loadData() -> (tf.data.Dataset, tf.data.Dataset):

    ### Load data and distribute
    photos = loadTFRecImages("photo_tfrec")
    style = loadTFRecImages("monet_tfrec")

    return photos, style

class GAN:
    def __init__(self, gan_model: str):

        ### Strategy scope
        with strategy.scope():

            if gan_model == 'small':
                self.model = SmallGAN()
                self.model.compile ()

            ### Siamese GAN
            elif gan_model == 'siamese':

                ## Generator/discriminator
                genStyleFn = Generator([*IMAGE_SIZE, 3], training = True)
                discStyleFn = Discriminator([*IMAGE_SIZE, 3])

                ## Losses
                euclProxLossFn = euclideanImageProximalLoss()
                euclDistLossFn = euclideanImageDistalLoss(0.5)
                cosSimPairLossFn = cosineSimilarityImagePairLoss()

                ## Optimizers
                genStyleOptimFn = adamOptimizer()
                discStyleOptimFn = adamOptimizer()

                # Model
                self.model = SiameseGAN(
                    genStyleFn, discStyleFn,
                    euclProxLossFn, euclDistLossFn, cosSimPairLossFn,
                    genStyleOptimFn, discStyleOptimFn)
                self.model.compile()

            ### Cycle GAN
            elif gan_model == 'cyclic':

                ## Generator/discriminator
                genPhotoFn = Generator([*IMAGE_SIZE, 3], training = True)
                genStyleFn = Generator([*IMAGE_SIZE, 3], training = True)
                discPhotoFn = Discriminator([*IMAGE_SIZE, 3])
                discStyleFn = Discriminator([*IMAGE_SIZE, 3])

                ## Losses
                cycleLossFn = identityLoss(5.)
                genLossFn = generatorLoss()
                discLossFn = discriminatorLoss()
                identLossFn = identityLoss(10.)

                ## Optimizers
                genPhotoOptimFn = adamOptimizer()
                genStyleOptimFn = adamOptimizer()
                discPhotoOptimFn = adamOptimizer()
                discStyleOptimFn = adamOptimizer()

                ## Model
                self.model = CycleGAN(
                                 genPhotoFn, genStyleFn, discPhotoFn, discStyleFn,
                                 cycleLossFn, genLossFn, discLossFn, identLossFn,
                                 genPhotoOptimFn, genStyleOptimFn, discPhotoOptimFn, discStyleOptimFn)
                self.model.compile()

    def train(self, data: tf.data.Dataset, epochs=EPOCHS) -> None:
        self.model.fit(data, epochs=epochs, verbose=1)

def plotImages(photos: tf.data.Dataset, n: int):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(n, 1, figsize=(24, 24))
    for i, img in enumerate(photos.take(n)):
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
        ax[i].imshow(img)
        ax[i].axis("off")
    plt.show()

if __name__ == '__main__':
    photos, style = loadData()
    photos = photos.batch(1)
    photos_s = photos.shuffle(BUFFER_SIZE)
    # plotImages(photos, 3)
    # plotImages(photos_s, 3)
    style = style.batch(1)
    gan = GAN('siamese')
    gan.train(tf.data.Dataset.zip((photos, photos_s, style)), epochs=1)
    # gan.savePredictions(photos)
    # from ImageLoader import plotImageList
    # plotImageList(styles)



