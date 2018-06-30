# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 이용한 간단한 Transfer Learning (TensorBoard 포함)
This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.
The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.
Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:
~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg
The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:
```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```
Or, if you have a pip installation of tensorflow, `retrain.py` can be run
without bazel:
```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos
```
You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.
This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.
To use with TensorBoard:
By default, this script will log summaries to /tmp/retrain_logs directory
Visualize the summaries with this command:
tensorboard --logdir /tmp/retrain_logs
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


class TrainMetaData:
    def __init__(self, base_dir='Files/Training'):
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        self.bottleneck_dir = os.path.join(base_dir, 'bottleneck')
        self.image_dir = os.path.join(base_dir, 'photos')
        self.final_tensor_name = 'final_result'
        self.eval_step_interval = 10
        self.flip_left_right = False
        self.how_many_training_steps = 1000
        self.learning_rate = 0.01
        self.model_dir = os.path.join(base_dir, 'imagenet')
        self.output_graph = os.path.join(base_dir, 'output_graph.pb')
        self.output_labels = os.path.join(base_dir, 'output_labels.txt')
        self.print_misclassified_test_images = False
        self.random_brightness = 0
        self.random_crop = 0
        self.random_scale = 0
        self.summaries_dir = os.path.join(base_dir, 'retrain_logs')
        self.test_batch_size = -1
        self.testing_percentage = 10
        self.train_batch_size = 100
        self.validation_batch_size = 100
        self.validation_percentage = 10
        self.graph_path = os.path.join(base_dir, 'output_graph.pb')  # 읽어들일 graph 파일 경로
        self.label_path = os.path.join(base_dir, 'output_labels.txt')  # 읽어들일 labels 파일 경로


class FaceClassification:
    def __init__(self):
        self.FLAGS = TrainMetaData()

        # 모든
        # 파라미터들은 특정한 모델 architecture와 묶여(tied) 있다.
        self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        self.BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
        self.BOTTLENECK_TENSOR_SIZE = 2048
        self.MODEL_INPUT_WIDTH = 299
        self.MODEL_INPUT_HEIGHT = 299
        self.MODEL_INPUT_DEPTH = 3
        self.JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
        self.RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
        self.MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


    def create_image_lists(self, image_dir, testing_percentage, validation_percentage):
        """file system으로부터 training 이미지들의 list를 만든다.
        이미지 디렉토리로부터 sub folder들을 분석하고, 그들을 training, testing, validation sets으로 나눈다.
        그리고 각각의 label을 위한 이미지 list와 그들의 경로(path)를 나타내는 자료구조(data structure)를 반환한다.
        인수들(Args):
          image_dir: 이미지들의 subfolder들을 포함한 folder의 String path.
          testing_percentage: 전체 이미지중 테스트를 위해 사용되는 비율을 나타내는 Integer.
          validation_percentage: 전체 이미지중 validation을 위해 사용되는 비율을 나타내는 Integer.
        반환값들(Returns):
          각각의 label subfolder를 위한 enrtry를 포함한 dictionary A dictionary
          (각각의 label에서 이미지드릉ㄴ training, testing, validation sets으로 나뉘어져 있다.)
        """
        if not gfile.Exists(image_dir):
            print("Image directory '" + image_dir + "' not found.")
            return None
        result = {}
        sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
        # root directory는 처음에 온다. 따라서 이를 skip한다.
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == image_dir:
                continue
            print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                print('No files found')
                continue
            if len(file_list) < 20:
                print('WARNING: Folder has less than 20 images, which may cause issues.')
            elif len(file_list) > self.MAX_NUM_IMAGES_PER_CLASS:
                print('WARNING: Folder {} has more than {} images. Some images will '
                      'never be selected.'.format(dir_name, self.MAX_NUM_IMAGES_PER_CLASS))
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                # 어떤 이미지로 리스트를 만들지 결정할때 파일 이름에 "_nohash_"가 포함되어 있으면 이를 무시할 수 있다.
                # 이를 이용해서, 데이터셋을 만드는 사람은 서로 비슷한 사진들을 grouping할 수있다.
                # 예를 들어, plant disease를 데이터셋을 만들기 위해서, 여러 장의 같은 잎사귀(leaf)를 grouping할 수 있다.
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                # 이는 일종의 마법처럼 보일 수 있다. 하지만, 우리는 이 파일이 training sets로 갈지, testing sets로 갈지, validation sets로 갈지 결정해야만 한다.
                # 그리고 우리는 더많은 파일들이 추가되더라도, 같은 set에 이미 존재하는 파일들이 유지되길 원한다.
                # 그렇게 하기 위해서는, 우리는 파일 이름 그자체로부터 결정하는 안정적인 방법이 있어야만 한다.
                # 따라서, 우리는 파일 이름을 hash하고, 이를 이를 할당하는데 사용하는 확률을 결정하는데 사용한다.
                hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                                    (self.MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                   (100.0 / self.MAX_NUM_IMAGES_PER_CLASS))
                if percentage_hash < validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)
            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }
        return result

    def get_image_path(self, image_lists, label_name, index, image_dir, category):
        """"주어진 index에 대한 이미지 경로(path)를 리턴한다.
        인수들(Args):
          image_lists: 각각의 label에 대한 training image들의 Dictionary.
          label_name: 우리가 얻고자하는 이미지의 Label string.
          index: 우리가 얻고자하는 이미지의 Int offset. 이는 레이블에 대한 가능한 이미지의 개수에 따라 moduloed 될 것이다.
          따라서 임의의 큰값이 될 수도 있다.
          image_dir: training 이미지들의 subfolder들을 포함하고 있는 Root folder string
          category: training, testing, 또는 validation sets으로부터 이미지에 pull할 Name string
        반환값(Returns):
          요청된 파라미터들이 만나게 될 이미지에 대한 파일 시스템 경로(file system path) string
        """
        if label_name not in image_lists:
            tf.logging.fatal('Label does not exist %s.', label_name)
        label_lists = image_lists[label_name]
        if category not in label_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = label_lists[category]
        if not category_list:
            tf.logging.fatal('Label %s has no images in the category %s.',
                             label_name, category)
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        full_path = os.path.join(image_dir, sub_dir, base_name)
        return full_path

    def get_bottleneck_path(self, image_lists, label_name, index, bottleneck_dir,
                            category):
        """"Returns a path to a bottleneck file for a label at the given index.
        Args:
          image_lists: Dictionary of training images for each label.
          label_name: Label string we want to get an image for.
          index: Integer offset of the image we want. This will be moduloed by the
          available number of images for the label, so it can be arbitrarily large.
          bottleneck_dir: Folder string holding cached files of bottleneck values.
          category: Name string of set to pull images from - training, testing, or
          validation.
        Returns:
          File system path string to an image that meets the requested parameters.
        """
        return self.get_image_path(image_lists, label_name, index, bottleneck_dir,
                                   category) + '.txt'

    def create_inception_graph(self):
        """"Creates a graph from saved GraphDef file and returns a Graph object.
        Returns:
          Graph holding the trained Inception network, and various tensors we'll be
          manipulating.
        """
        with tf.Graph().as_default() as graph:
            model_filename = os.path.join(
                self.FLAGS.model_dir, 'classify_image_graph_def.pb')
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                        self.BOTTLENECK_TENSOR_NAME, self.JPEG_DATA_TENSOR_NAME,
                        self.RESIZED_INPUT_TENSOR_NAME]))
        return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor,
                                bottleneck_tensor):
        """Runs inference on an image to extract the 'bottleneck' summary layer.
        Args:
          sess: Current active TensorFlow Session.
          image_data: String of raw JPEG data.
          image_data_tensor: Input data layer in the graph.
          bottleneck_tensor: Layer before the final softmax.
        Returns:
          Numpy array of bottleneck values.
        """
        bottleneck_values = sess.run(
            bottleneck_tensor,
            {image_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def maybe_download_and_extract(self):
        """Download and extract model tar file.
        If the pretrained model we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a directory.
        """
        dest_directory = self.FLAGS.model_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename,
                                  float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(self.DATA_URL,
                                                     filepath,
                                                     _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def ensure_dir_exists(self, dir_name):
        """Makes sure the folder exists on disk.
        Args:
          dir_name: Path string to the folder we want to create.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def write_list_of_floats_to_file(self, list_of_floats, file_path):
        """Writes a given list of floats to a binary file.
        Args:
          list_of_floats: List of floats we want to write to a file.
          file_path: Path to a file where list of floats will be stored.
        """

        s = struct.pack('d' * self.BOTTLENECK_TENSOR_SIZE, *list_of_floats)
        with open(file_path, 'wb') as f:
            f.write(s)

    def read_list_of_floats_from_file(self, file_path):
        """Reads list of floats from a given file.
        Args:
          file_path: Path to a file where list of floats was stored.
        Returns:
          Array of bottleneck values (list of floats).
        """

        with open(file_path, 'rb') as f:
            s = struct.unpack('d' * self.BOTTLENECK_TENSOR_SIZE, f.read())
            return list(s)

    bottleneck_path_2_bottleneck_values = {}

    def create_bottleneck_file(self, bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor):
        """Create a single bottleneck file."""
        print('Creating bottleneck at ' + bottleneck_path)
        image_path = self.get_image_path(image_lists, label_name, index,
                                         image_dir, category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        try:
            bottleneck_values = self.run_bottleneck_on_image(
                sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        except:
            raise RuntimeError('Error during processing file %s' % image_path)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    def get_or_create_bottleneck(self, sess, image_lists, label_name, index, image_dir,
                                 category, bottleneck_dir, jpeg_data_tensor,
                                 bottleneck_tensor):
        """Retrieves or calculates bottleneck values for an image.
        If a cached version of the bottleneck data exists on-disk, return that,
        otherwise calculate the data and save it to disk for future use.
        Args:
          sess: The current active TensorFlow Session.
          image_lists: Dictionary of training images for each label.
          label_name: Label string we want to get an image for.
          index: Integer offset of the image we want. This will be modulo-ed by the
          available number of images for the label, so it can be arbitrarily large.
          image_dir: Root folder string  of the subfolders containing the training
          images.
          category: Name string of which  set to pull images from - training, testing,
          or validation.
          bottleneck_dir: Folder string holding cached files of bottleneck values.
          jpeg_data_tensor: The tensor to feed loaded jpeg data into.
          bottleneck_tensor: The output tensor for the bottleneck values.
        Returns:
          Numpy array of values produced by the bottleneck layer for the image.
        """
        label_lists = image_lists[label_name]
        sub_dir = label_lists['dir']
        sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
        self.ensure_dir_exists(sub_dir_path)
        bottleneck_path = self.get_bottleneck_path(image_lists, label_name, index,
                                                   bottleneck_dir, category)
        if not os.path.exists(bottleneck_path):
            self.create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                                        image_dir, category, sess, jpeg_data_tensor,
                                        bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        did_hit_error = False
        try:
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        except ValueError:
            print('Invalid float found, recreating bottleneck')
            did_hit_error = True
        if did_hit_error:
            self.create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                                        image_dir, category, sess, jpeg_data_tensor,
                                        bottleneck_tensor)
            with open(bottleneck_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
            # Allow exceptions to propagate here, since they shouldn't happen after a
            # fresh creation
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        return bottleneck_values

    def cache_bottlenecks(self, sess, image_lists, image_dir, bottleneck_dir,
                          jpeg_data_tensor, bottleneck_tensor):
        """Ensures all the training, testing, and validation bottlenecks are cached.
        Because we're likely to read the same image multiple times (if there are no
        distortions applied during training) it can speed things up a lot if we
        calculate the bottleneck layer values once for each image during
        preprocessing, and then just read those cached values repeatedly during
        training. Here we go through all the images we've found, calculate those
        values, and save them off.
        Args:
          sess: The current active TensorFlow Session.
          image_lists: Dictionary of training images for each label.
          image_dir: Root folder string of the subfolders containing the training
          images.
          bottleneck_dir: Folder string holding cached files of bottleneck values.
          jpeg_data_tensor: Input tensor for jpeg data from file.
          bottleneck_tensor: The penultimate output layer of the graph.
        Returns:
          Nothing.
        """
        how_many_bottlenecks = 0
        self.ensure_dir_exists(bottleneck_dir)
        for label_name, label_lists in image_lists.items():
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, unused_base_name in enumerate(category_list):
                    self.get_or_create_bottleneck(sess, image_lists, label_name, index,
                                                  image_dir, category, bottleneck_dir,
                                                  jpeg_data_tensor, bottleneck_tensor)

                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        print(str(how_many_bottlenecks) + ' bottleneck files created.')

    def get_random_cached_bottlenecks(self, sess, image_lists, how_many, category,
                                      bottleneck_dir, image_dir, jpeg_data_tensor,
                                      bottleneck_tensor):
        """Retrieves bottleneck values for cached images.
        If no distortions are being applied, this function can retrieve the cached
        bottleneck values directly from disk for images. It picks a random set of
        images from the specified category.
        Args:
          sess: Current TensorFlow Session.
          image_lists: Dictionary of training images for each label.
          how_many: If positive, a random sample of this size will be chosen.
          If negative, all bottlenecks will be retrieved.
          category: Name string of which set to pull from - training, testing, or
          validation.
          bottleneck_dir: Folder string holding cached files of bottleneck values.
          image_dir: Root folder string of the subfolders containing the training
          images.
          jpeg_data_tensor: The layer to feed jpeg image data into.
          bottleneck_tensor: The bottleneck output layer of the CNN graph.
        Returns:
          List of bottleneck arrays, their corresponding ground truths, and the
          relevant filenames.
        """
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        filenames = []
        if how_many >= 0:
            # Retrieve a random sample of bottlenecks.
            for unused_i in range(how_many):
                label_index = random.randrange(class_count)
                label_name = list(image_lists.keys())[label_index]
                image_index = random.randrange(self.MAX_NUM_IMAGES_PER_CLASS + 1)
                image_name = self.get_image_path(image_lists, label_name, image_index,
                                                 image_dir, category)
                bottleneck = self.get_or_create_bottleneck(sess, image_lists, label_name,
                                                           image_index, image_dir, category,
                                                           bottleneck_dir, jpeg_data_tensor,
                                                           bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
        else:
            # Retrieve all bottlenecks.
            for label_index, label_name in enumerate(image_lists.keys()):
                for image_index, image_name in enumerate(
                        image_lists[label_name][category]):
                    image_name = self.get_image_path(image_lists, label_name, image_index,
                                                     image_dir, category)
                    bottleneck = self.get_or_create_bottleneck(sess, image_lists, label_name,
                                                               image_index, image_dir, category,
                                                               bottleneck_dir, jpeg_data_tensor,
                                                               bottleneck_tensor)
                    ground_truth = np.zeros(class_count, dtype=np.float32)
                    ground_truth[label_index] = 1.0
                    bottlenecks.append(bottleneck)
                    ground_truths.append(ground_truth)
                    filenames.append(image_name)
        return bottlenecks, ground_truths, filenames

    def get_random_distorted_bottlenecks(self,
                                         sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
                                         distorted_image, resized_input_tensor, bottleneck_tensor):
        """Retrieves bottleneck values for training images, after distortions.
        If we're training with distortions like crops, scales, or flips, we have to
        recalculate the full model for every image, and so we can't use cached
        bottleneck values. Instead we find random images for the requested category,
        run them through the distortion graph, and then the full graph to get the
        bottleneck results for each.
        Args:
          sess: Current TensorFlow Session.
          image_lists: Dictionary of training images for each label.
          how_many: The integer number of bottleneck values to return.
          category: Name string of which set of images to fetch - training, testing,
          or validation.
          image_dir: Root folder string of the subfolders containing the training
          images.
          input_jpeg_tensor: The input layer we feed the image data to.
          distorted_image: The output node of the distortion graph.
          resized_input_tensor: The input node of the recognition graph.
          bottleneck_tensor: The bottleneck output layer of the CNN graph.
        Returns:
          List of bottleneck arrays and their corresponding ground truths.
        """
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(self.MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(image_lists, label_name, image_index, image_dir,
                                             category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            # Note that we materialize the distorted_image_data as a numpy array before
            # sending running inference on the image. This involves 2 memory copies and
            # might be optimized in other implementations.
            distorted_image_data = sess.run(distorted_image,
                                            {input_jpeg_tensor: jpeg_data})
            bottleneck = self.run_bottleneck_on_image(sess, distorted_image_data,
                                                      resized_input_tensor,
                                                      bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
        return bottlenecks, ground_truths

    def should_distort_images(self, flip_left_right, random_crop, random_scale,
                              random_brightness):
        """Whether any distortions are enabled, from the input flags.
        Args:
          flip_left_right: Boolean whether to randomly mirror images horizontally.
          random_crop: Integer percentage setting the total margin used around the
          crop box.
          random_scale: Integer percentage of how much to vary the scale by.
          random_brightness: Integer range to randomly multiply the pixel values by.
        Returns:
          Boolean value indicating whether any distortions should be applied.
        """
        return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
                (random_brightness != 0))

    def add_input_distortions(self, flip_left_right, random_crop, random_scale,
                              random_brightness):
        """Creates the operations to apply the specified distortions.
        During training it can help to improve the results if we run the images
        through simple distortions like crops, scales, and flips. These reflect the
        kind of variations we expect in the real world, and so can help train the
        model to cope with natural data more effectively. Here we take the supplied
        parameters and construct a network of operations to apply them to an image.
        Cropping
        ~~~~~~~~
        Cropping is done by placing a bounding box at a random position in the full
        image. The cropping parameter controls the size of that box relative to the
        input image. If it's zero, then the box is the same size as the input and no
        cropping is performed. If the value is 50%, then the crop box will be half the
        width and height of the input. In a diagram it looks like this:
        <       width         >
        +---------------------+
        |                     |
        |   width - crop%     |
        |    <      >         |
        |    +------+         |
        |    |      |         |
        |    |      |         |
        |    |      |         |
        |    +------+         |
        |                     |
        |                     |
        +---------------------+
        Scaling
        ~~~~~~~
        Scaling is a lot like cropping, except that the bounding box is always
        centered and its size varies randomly within the given range. For example if
        the scale percentage is zero, then the bounding box is the same size as the
        input and no scaling is applied. If it's 50%, then the bounding box will be in
        a random range between half the width and height and full size.
        Args:
          flip_left_right: Boolean whether to randomly mirror images horizontally.
          random_crop: Integer percentage setting the total margin used around the
          crop box.
          random_scale: Integer percentage of how much to vary the scale by.
          random_brightness: Integer range to randomly multiply the pixel values by.
          graph.
        Returns:
          The jpeg input layer and the distorted result tensor.
        """

        jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=self.MODEL_INPUT_DEPTH)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                               minval=1.0,
                                               maxval=resize_scale)
        scale_value = tf.multiply(margin_scale_value, resize_scale_value)
        precrop_width = tf.multiply(scale_value, self.MODEL_INPUT_WIDTH)
        precrop_height = tf.multiply(scale_value, self.MODEL_INPUT_HEIGHT)
        precrop_shape = tf.stack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
        precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                    precrop_shape_as_int)
        precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
        cropped_image = tf.random_crop(precropped_image_3d,
                                       [self.MODEL_INPUT_HEIGHT, self.MODEL_INPUT_WIDTH,
                                        self.MODEL_INPUT_DEPTH])
        if flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                             minval=brightness_min,
                                             maxval=brightness_max)
        brightened_image = tf.multiply(flipped_image, brightness_value)
        distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
        return jpeg_data, distort_result

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor):
        """Adds a new softmax and fully-connected layer for training.
        We need to retrain the top layer to identify our new classes, so this function
        adds the right operations to the graph, along with some variables to hold the
        weights, and then sets up all the gradients for the backward pass.
        The set up for the softmax and fully-connected layers is based on:
        https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
        Args:
          class_count: Integer of how many categories of things we're trying to
          recognize.
          final_tensor_name: Name string for the new final node that produces results.
          bottleneck_tensor: The output of the main CNN graph.
        Returns:
          The tensors for the training and cross entropy results, and tensors for the
          bottleneck input and ground truth input.
        """
        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor, shape=[None, self.BOTTLENECK_TENSOR_SIZE],
                name='BottleneckInputPlaceholder')

            ground_truth_input = tf.placeholder(tf.float32,
                                                [None, class_count],
                                                name='GroundTruthInput')

        # Organizing the following ops as `final_training_ops` so they're easier
        # to see in TensorBoard
        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value = tf.truncated_normal([self.BOTTLENECK_TENSOR_SIZE, class_count],
                                                    stddev=0.001)

                layer_weights = tf.Variable(initial_value, name='final_weights')

                self.variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                self.variable_summaries(layer_biases)
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations', logits)

        final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
        tf.summary.histogram('activations', final_tensor)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate)
            train_step = optimizer.minimize(cross_entropy_mean)

        return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor)

    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        """Inserts the operations we need to evaluate the accuracy of our results.
        Args:
          result_tensor: The new final node that produces results.
          ground_truth_tensor: The node we feed ground truth data
          into.
        Returns:
          Tuple of (evaluation step, prediction).
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(result_tensor, 1)
                correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step, prediction

    def start_training(self):
        # tf.app.run(main=self.main, argv=[sys.argv[0]] + self.unparsed)
        # TensorBoard의 summaries를 write할 directory를 설정한다.

        if tf.gfile.Exists(self.FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(self.FLAGS.summaries_dir)
        tf.gfile.MakeDirs(self.FLAGS.summaries_dir)

        # pre-trained graph를 생성한다.
        self.maybe_download_and_extract()
        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
                self.create_inception_graph())

        # 폴더 구조를 살펴보고, 모든 이미지에 대한 lists를 생성한다.
        image_lists = self.create_image_lists(self.FLAGS.image_dir, self.FLAGS.testing_percentage,
                                                      self.FLAGS.validation_percentage)
        class_count = len(image_lists.keys())
        if class_count == 0:
            error_msg = 'No valid folders of images found at ' + self.FLAGS.image_dir
            print(error_msg)
        if class_count == 1:
            error_msg = 'Only one valid folder of ' \
                    'images found at ' + self.FLAGS.image_dir + ' - multiple classes are needed for classification.'
            print(error_msg)

                        # 커맨드라인 flag에 distortion에 관련된 설정이 있으면 distortion들을 적용한다.
        do_distort_images = self.should_distort_images(
            self.FLAGS.flip_left_right, self.FLAGS.random_crop, self.FLAGS.random_scale,
            self.FLAGS.random_brightness)

        with tf.Session(graph=graph) as sess:

            if do_distort_images:
                # 우리는 distortion들을 적용할것이다. 따라서 필요한 연산들(operations)을 설정한다.
                (distorted_jpeg_data_tensor,
                 distorted_image_tensor) = self.add_input_distortions(
                    self.FLAGS.flip_left_right, self.FLAGS.random_crop,
                    self.FLAGS.random_scale, self.FLAGS.random_brightness)
            else:
                # 우리는 계산된 'bottleneck' 이미지 summaries를 가지고 있다.
                # 이를 disk에 캐싱(caching)할 것이다.
                self.cache_bottlenecks(sess, image_lists, self.FLAGS.image_dir,
                                       self.FLAGS.bottleneck_dir, jpeg_data_tensor,
                                       bottleneck_tensor)

            # 우리가 학습시킬(training) 새로운 layer를 추가한다.
            (train_step, cross_entropy, bottleneck_input, ground_truth_input,
             final_tensor) = self.add_final_training_ops(len(image_lists.keys()),
                                                         self.FLAGS.final_tensor_name,
                                                         bottleneck_tensor)

            # 우리의 새로운 layer의 정확도를 평가(evalute)하기 위한 새로운 operation들을 생성한다.
            evaluation_step, prediction = self.add_evaluation_step(
                final_tensor, ground_truth_input)

            # 모든 summaries를 합치고(merge) summaries_dir에 쓴다.(write)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir + '/train',
                                                 sess.graph)

            validation_writer = tf.summary.FileWriter(
                self.FLAGS.summaries_dir + '/validation')

            # 우리의 모든 가중치들(weights)과 그들의 초기값들을 설정한다.
            init = tf.global_variables_initializer()
            sess.run(init)

            # 커맨드 라인에서 지정한 횟수만큼 학습을 진행한다.
            for i in range(self.FLAGS.how_many_training_steps):
                # bottleneck 값들의 batch를 얻는다. 이는 매번 distortion을 적용하고 계산하거나,
                # disk에 저장된 chache로부터 얻을 수 있다.
                if do_distort_images:
                    (train_bottlenecks,
                     train_ground_truth) = self.get_random_distorted_bottlenecks(
                        sess, image_lists, self.FLAGS.train_batch_size, 'training',
                        self.FLAGS.image_dir, distorted_jpeg_data_tensor,
                        distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
                else:
                    (train_bottlenecks,
                     train_ground_truth, _) = self.get_random_cached_bottlenecks(
                        sess, image_lists, self.FLAGS.train_batch_size, 'training',
                        self.FLAGS.bottleneck_dir, self.FLAGS.image_dir, jpeg_data_tensor,
                        bottleneck_tensor)
                # grpah에 bottleneck과 ground truth를 feed하고, training step을 진행한다.
                # TensorBoard를 위한 'merged' op을 이용해서 training summaries을 capture한다.

                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                train_writer.add_summary(train_summary, i)

                # 일정 step마다 graph의 training이 얼마나 잘 되고 있는지 출력한다.
                is_last_step = (i + 1 == self.FLAGS.how_many_training_steps)
                if (i % self.FLAGS.eval_step_interval) == 0 or is_last_step:
                    train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth})
                    print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                                    train_accuracy * 100))
                    print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                               cross_entropy_value))
                    validation_bottlenecks, validation_ground_truth, _ = (
                        self.get_random_cached_bottlenecks(
                            sess, image_lists, self.FLAGS.validation_batch_size, 'validation',
                            self.FLAGS.bottleneck_dir, self.FLAGS.image_dir, jpeg_data_tensor,
                            bottleneck_tensor))
                    # validation step을 진행한다.
                    # TensorBoard를 위한 'merged' op을 이용해서 training summaries을 capture한다.
                    validation_summary, validation_accuracy = sess.run(
                        [merged, evaluation_step],
                        feed_dict={bottleneck_input: validation_bottlenecks,
                                   ground_truth_input: validation_ground_truth})
                    validation_writer.add_summary(validation_summary, i)
                    print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                          (datetime.now(), i, validation_accuracy * 100,
                           len(validation_bottlenecks)))

            # 트레이닝 과정이 모두 끝났다.
            # 따라서 이전에 보지 못했던 이미지를 통해 마지막 test 평가(evalution)을 진행한다.
            test_bottlenecks, test_ground_truth, test_filenames = (
                self.get_random_cached_bottlenecks(sess, image_lists, self.FLAGS.test_batch_size,
                                                   'testing', self.FLAGS.bottleneck_dir,
                                                   self.FLAGS.image_dir, jpeg_data_tensor,
                                                   bottleneck_tensor))
            test_accuracy, predictions = sess.run(
                [evaluation_step, prediction],
                feed_dict={bottleneck_input: test_bottlenecks,
                           ground_truth_input: test_ground_truth})
            print('Final test accuracy = %.1f%% (N=%d)' % (
                test_accuracy * 100, len(test_bottlenecks)))

            if self.FLAGS.print_misclassified_test_images:
                print('=== MISCLASSIFIED TEST IMAGES ===')
                for i, test_filename in enumerate(test_filenames):
                    if predictions[i] != test_ground_truth[i].argmax():
                        print('%70s  %s' % (test_filename,
                                            list(image_lists.keys())[predictions[i]]))

            # 학습된 graph와 weights들을 포함한 labels를 쓴다.(write)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), [self.FLAGS.final_tensor_name])
            with gfile.FastGFile(self.FLAGS.output_graph, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            with gfile.FastGFile(self.FLAGS.output_labels, 'w') as f:
                f.write('\n'.join(image_lists.keys()) + '\n')

    def create_graph(self, graph_path):
        """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
        # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
        with tf.gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def get_accuracy(self):
        answer = None
        folder_path = os.path.join('Files', 'Training')
        image_path = os.path.join(folder_path, 'test.jpg')  # , file_name)
        label_path = os.path.join(folder_path, 'output_labels.txt')
        graph_path = os.path.join(folder_path, 'output_graph.pb')
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
            return answer

        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
        self.create_graph(graph_path)

        with tf.Session() as sess:

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
            f = open(label_path, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))

            uid_label, accuracy = labels[top_k[0]], predictions[top_k[0]]
            uid_label = uid_label.split("'")[1].split('\\')[0]

            dir_path = os.path.join(folder_path, 'Patientphotos')
            dir_list = os.listdir(dir_path)
            for dir in dir_list:
                if dir.lower() == uid_label.lower():
                    uid_label = dir
                    break
            return uid_label, accuracy

    def get_result(self):
        uid, accuracy = self.get_accuracy()
        return uid, accuracy