import unittest
import matplotlib.pyplot as plt
import tensorflow as tf
from ImageLoader import loadTFRecImages, loadJpgImages

WINDOW_ROWS = 16
WINDOW_COLUMNS = 16
STRIDE_ROWS = 16
STRIDE_COLUMNS = 16

def mapToPatches(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Slice images to patches of a given size - using TensorFlow extract patches op
    :param images:  An input of batch images to slice to patches
    :return:
    """
    dataset = dataset.map(lambda image: tf.image.extract_patches(images=image,
                                    sizes=[1, WINDOW_ROWS, WINDOW_COLUMNS, 1],
                                    strides=[1, STRIDE_ROWS, STRIDE_COLUMNS, 1],
                                    rates=[1, 1, 1, 1],
                                    padding='VALID'))
    return dataset

def plotPatches(dataset: tf.data.Dataset) -> None:
    patches = next(dataset.as_numpy_iterator())
    rows = patches.shape[1]
    columns = patches.shape[2]
    for iRow in range(rows):
        for iColumn in range(columns):
            plt.subplot(rows, columns, 1 + iColumn + iRow * columns)
            patch = patches[0, iRow, iColumn, :].reshape(WINDOW_ROWS, WINDOW_COLUMNS, -1)
            plt.imshow(patch / 255.)
            plt.axis('off')
            plt.subplots_adjust(wspace=0.01, hspace=0.1)
    plt.show()

class TestWindowing(unittest.TestCase):
    def setUp(self):
        pass

    def test_windowing(self):
        dataset = loadTFRecImages("photo_tfrec")
        # dataset = loadJpgImages("photo_jpg")
        patches = mapToPatches(dataset)
        plotPatches(patches)

if __name__ == '__main__':
    unittest.main()
