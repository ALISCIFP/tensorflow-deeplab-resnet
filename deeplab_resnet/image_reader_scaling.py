import math

import SimpleITK as sitk
import numpy as np
import tensorflow as tf


def read_nii_and_image_scaling(img_fname, label_fname):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    output_spacing = tf.add(tf.multiply(tf.random_uniform([3], 0, 1.0, dtype=tf.float64),
                                        tf.constant([1 - .55, 1 - .55, 6 - .45], dtype=tf.float64)),
                            tf.constant([.55, .55, .45], dtype=tf.float64))

    img, input_spacing = tf.py_func(threed_rescale_bilinear, [img_fname, output_spacing], [tf.float32, tf.float64])
    label = tf.py_func(threed_rescale_nn, [label_fname, output_spacing, input_spacing], tf.uint8)

    img = tf.subtract(img, tf.reduce_mean(img))
    img = tf.divide(img, tf.reduce_max(img) - tf.reduce_min(img))
    return img, label


def threed_rescale_nn(orig_image_fname, output_spacing, input_spacing):
    img_orig = sitk.ReadImage(orig_image_fname)
    origin = img_orig.GetOrigin()
    direction = img_orig.GetDirection()
    size = img_orig.GetSize()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetOutputSpacing(output_spacing.tolist())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    output_size = [int(math.ceil(input_spacing[0] * (size[0] - 1) / output_spacing[0]) + 1),
                   int(math.ceil(input_spacing[1] * (size[1] - 1) / output_spacing[1]) + 1),
                   int(math.ceil(input_spacing[2] * (size[2] - 1) / output_spacing[2]) + 1)]
    resampler.SetSize(output_size)

    output = sitk.GetArrayFromImage(resampler.Execute(img_orig)).astype(np.uint8).transpose()
    # output = np.pad(output, ((0, 0), (0, 0), (5, 6)), 'constant')

    return output


def threed_rescale_bilinear(orig_image_fname, output_spacing):
    img_orig = sitk.ReadImage(orig_image_fname)
    origin = img_orig.GetOrigin()
    direction = img_orig.GetDirection()
    spacing = img_orig.GetSpacing()
    size = img_orig.GetSize()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetOutputSpacing(output_spacing.tolist())
    resampler.SetInterpolator(sitk.sitkLinear)

    output_size = [int(math.ceil(spacing[0] * (size[0] - 1) / output_spacing[0]) + 1),
                   int(math.ceil(spacing[1] * (size[1] - 1) / output_spacing[1]) + 1),
                   int(math.ceil(spacing[2] * (size[2] - 1) / output_spacing[2]) + 1)]
    resampler.SetSize(output_size)

    output = sitk.GetArrayFromImage(resampler.Execute(img_orig)).astype(np.float32).transpose()
    output = np.clip(output, -200, 200)
    # output = np.pad(output, ((0, 0), (0, 0), (6, 7)), 'constant')

    return output, np.array(spacing, dtype=np.float64)


def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_random = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
    mirror = tf.less(tf.stack([1.0, distort_random[0], distort_random[1]]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label


def random_crop_and_pad_image_and_labels(img, label, crop_h, crop_w, t, w):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    # 1: pad 2: crop
    image_shape = tf.shape(img)

    def pad_image_x(img, crop_x_size, img_x_size):
        crop_x_size = tf.cast(crop_x_size, dtype=tf.float32)
        img_x_size = tf.cast(img_x_size, dtype=tf.float32)
        return tf.pad(img, [[tf.cast(tf.ceil((crop_x_size - img_x_size) / 2), dtype=tf.int32),
                             tf.cast(tf.ceil((crop_x_size - img_x_size) / 2), dtype=tf.int32)], [0, 0],
                            [0, 0]], mode="CONSTANT")

    def no_pad_image_x(img):
        return img

    x_need_pad_flag = tf.greater(crop_h, image_shape[0])
    img = tf.cond(x_need_pad_flag, lambda: pad_image_x(img, crop_h, image_shape[0]), lambda: no_pad_image_x(img))
    label = tf.cond(x_need_pad_flag, lambda: pad_image_x(label, crop_h, image_shape[0]), lambda: no_pad_image_x(label))

    def pad_image_y(img, crop_y_size, img_y_size):
        crop_y_size = tf.cast(crop_y_size, dtype=tf.float32)
        img_y_size = tf.cast(img_y_size, dtype=tf.float32)
        return tf.pad(img, [[0, 0], [tf.cast(tf.ceil((crop_y_size - img_y_size) / 2), dtype=tf.int32),
                                     tf.cast(tf.ceil((crop_y_size - img_y_size) / 2), dtype=tf.int32)],
                            [0, 0]], mode="CONSTANT")

    def no_pad_image_y(img):
        return img

    y_need_pad_flag = tf.greater(crop_w, image_shape[1])
    img = tf.cond(y_need_pad_flag, lambda: pad_image_y(img, crop_w, image_shape[1]), lambda: no_pad_image_y(img))
    label = tf.cond(y_need_pad_flag, lambda: pad_image_y(label, crop_w, image_shape[1]), lambda: no_pad_image_y(label))

    def pad_image_z(img, crop_z_size, img_z_size):
        crop_z_size = tf.cast(crop_z_size, dtype=tf.float32)
        img_z_size = tf.cast(img_z_size, dtype=tf.float32)
        return tf.pad(img, [[0, 0], [0, 0],
                            [tf.cast(tf.ceil((crop_z_size - img_z_size) / 2), dtype=tf.int32),
                             tf.cast(tf.ceil((crop_z_size - img_z_size) / 2), dtype=tf.int32)]],
                      mode="CONSTANT")

    def no_pad_image_z(img):
        return img

    z_need_pad_flag = tf.greater(14, image_shape[2])
    img = tf.cond(z_need_pad_flag, lambda: pad_image_z(img, 14, image_shape[2]), lambda: no_pad_image_z(img))
    label = tf.cond(z_need_pad_flag, lambda: pad_image_z(label, 14, image_shape[2]), lambda: no_pad_image_z(label))

    image_shape = tf.shape(img)

    crop_x_value = tf.clip_by_value(tf.cast(
        tf.round(tf.random_uniform([1], 0, tf.cast(tf.maximum(crop_h, image_shape[0]) - crop_h, dtype=tf.float32),
                                   dtype=tf.float32)), dtype=tf.int32), 0, tf.maximum(crop_h, image_shape[0]) - crop_h)
    crop_y_value = tf.clip_by_value(tf.cast(
        tf.round(tf.random_uniform([1], 0, tf.cast(tf.maximum(crop_w, image_shape[1]) - crop_w, dtype=tf.float32),
                                   dtype=tf.float32)), dtype=tf.int32), 0,
        tf.maximum(crop_w, image_shape[1]) - crop_w)
    crop_z_value = tf.clip_by_value(
        tf.cast(tf.round(tf.random_uniform([1], 6, tf.cast(tf.maximum(14, image_shape[2] - 7) - 14, dtype=tf.float32),
                                           dtype=tf.float32)), dtype=tf.int32), 0,
        tf.maximum(14, image_shape[2]) - 14)
    crop_z_value = crop_z_value + 6

    img_crop = img[tf.squeeze(crop_x_value):tf.squeeze(crop_x_value + crop_h),
               tf.squeeze(crop_y_value):tf.squeeze(crop_y_value + crop_w),
               tf.squeeze(crop_z_value - 6):tf.squeeze(crop_z_value + 8)]
    label_crop = label[tf.squeeze(crop_x_value):tf.squeeze(crop_x_value + crop_h),
                 tf.squeeze(crop_y_value):tf.squeeze(crop_y_value + crop_w),
                 tf.squeeze(crop_z_value - 5):tf.squeeze(crop_z_value + 7)]

    # 3 stack the image by channels
    img_list = []
    for i in range(1, 13):
        img_list.append(img_crop[:, :, i - 1: i + 2])

    img_crop = tf.concat(img_list, axis=-1)

    img_crop = tf.Print(img_crop, [t, w, tf.shape(img), tf.shape(label), tf.shape(img_crop), tf.shape(label_crop)])

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 36))
    label_crop.set_shape((crop_h, crop_w, 12))
    return img_crop, label_crop


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split('\t')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label,
                          img_mean):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """
    h, w = input_size

    # Randomly scale the images and labels.
    img, label = read_nii_and_image_scaling(input_queue[0], input_queue[1])

    # Randomly crops the images and labels.
    img, label = random_crop_and_pad_image_and_labels(img, label, h, w, input_queue[0], input_queue[1])

    # Randomly mirror the images and labels.
    img, label = image_mirroring(img, label)

    return img, label


class ImageReaderScaling(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size,
                 random_scale, random_mirror, ignore_label, img_mean, coord, shuffle=True, num_threads=4):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        self.shuffle = shuffle
        self.num_threads = num_threads

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=shuffle)  # not shuffling if it is val
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror,
                                                       ignore_label, img_mean)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''

        example_list = [(self.image, self.label) for _ in range(self.num_threads)]
        image_batch, label_batch = tf.train.batch_join(example_list, num_elements,
                                                       capacity=num_elements * self.num_threads)

        # image_batch, label_batch = tf.train.batch([self.image, self.label],
        #                                           num_elements)

        return image_batch, label_batch
