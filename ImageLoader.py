import unittest
import os
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from typing import List, Any
import tensorflow as tf
import numpy as np

DATA_PATH = "/Users/hollander/OneDrive/Documents/Career/ProjectNextGen/Kaggle/MonetStyleTransfer/Data/"
IMAGE_SIZE = [256, 256]
BATCH_SIZE = 64

def decodeImage(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1    # Decode image pixels to the range [-1, 1]
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def readTFRecord(example):
    tfrecordFormat = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecordFormat)
    image = decodeImage(example["image"])
    return image

def loadTFDataset(recordsToLoad: List[str]) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(recordsToLoad)
    dataset = dataset.map(readTFRecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def listOfRecordsToLoad(dir: str, required_file_extension: str) -> List[str]:
    path = os.walk(os.path.join(DATA_PATH, dir))
    recordsToLoad: List[str] = []
    for root, directories, files in path:
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension != required_file_extension:
                continue
            recordsToLoad.append(os.path.join(root, file))
    return recordsToLoad

def loadTFRecImages(dir: str) -> tf.data.Dataset:
    """
    Load all TF record images under subdirectory of data path
    :param dir: Subdirectory under data root path
    :return: A tensorflow dataset object
    """
    dataset = loadTFDataset(listOfRecordsToLoad(dir, ".tfrec"))
    return dataset

def castJpgType(example):
    image = (tf.cast(example, tf.float32) / 127.5) - 1  # Decode image pixels to the range [-1, 1]
    return image

def loadJpgImages(dir: str) -> tf.data.Dataset:
    """
    Load all jpg images under subdirectory of data path
    :param dir: Subdirectory under data root path
    :return: A tensorflow dataset object
    """
    recordsToLoad = listOfRecordsToLoad(dir, ".jpg")
    images = []
    for recordToLoad in recordsToLoad:
        images.append(mpimg.imread(recordToLoad))
    images = np.array(images)
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(castJpgType)
    dataset = dataset.shuffle(2048)
    return dataset

def plotImages(dataset: tf.data.Dataset, n: int = 1):
    plt.figure()
    image = next(dataset.as_numpy_iterator())
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(image[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

class TestImageLoader(unittest.TestCase):
    def setUp(self):
        pass

    def test_load(self):
        dataset = loadJpgImages("monet_jpg").batch(20)
        plotImages(dataset, n = 5)

    def test_loadTF(self):
        dataset = loadTFRecImages("monet_tfrec").batch(20)
        plotImages(dataset, n = 3)

if __name__ == '__main__':
    unittest.main()
