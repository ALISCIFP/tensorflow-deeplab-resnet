import os

import tensorflow as tf


def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label


def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 10])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 9))
    label_crop.set_shape((crop_h, crop_w, 1))
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

    img_contents = tf.read_file(input_queue[0])
    str_split1 = tf.string_split([input_queue[0]], delimiter="_")

    str_split2 = tf.string_split([str_split1.values[1]], delimiter=".")

    str_tail_num = tf.string_to_number(str_split2.values[0], tf.int32)
    f0 = tf.string_join([str_split1.values[0], '_', tf.as_string(str_tail_num - 3), '.', str_split2.values[1]])
    f2 = tf.string_join([str_split1.values[0], '_', tf.as_string(str_tail_num + 3), '.', str_split2.values[1]])

    # f0 = tf.Print(f0, [f0])
    # f2 = tf.Print(f2, [f2])
    label_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_png(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    def read_image(fname):

        image_contents = tf.read_file(fname)
        image = tf.image.decode_png(image_contents, channels=3)
        image_r, image_g, image_b = tf.split(axis=2, num_or_size_splits=3, value=image)
        image = tf.cast(tf.concat(axis=2, values=[image_b, image_g, image_r]), dtype=tf.float32)
        # Extract mean.
        image -= img_mean
        return image

    def fail(img, fname):
        img = tf.Print(img, [fname])
        return img

    f0_exists_flag, = tf.py_func(os.path.exists, [f0], [tf.bool])
    img0 = tf.cond(f0_exists_flag, lambda: read_image(f0), lambda: fail(img, f0))

    f2_exists_flag, = tf.py_func(os.path.exists, [f2], [tf.bool])
    img2 = tf.cond(f2_exists_flag, lambda: read_image(f2), lambda: fail(img, f2))

    label = tf.image.decode_png(label_contents, channels=1)
    img = tf.concat([img2, img, img0], axis=-1)[2:7]

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label


class ImageReader(object):
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

        if self.shuffle:
            example_list = [(self.image, self.label) for _ in range(self.num_threads)]
            image_batch, label_batch = tf.train.batch_join(example_list, num_elements,
                                                           capacity=num_elements * self.num_threads)
        else:
            image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                      num_elements)
        return image_batch, label_batch
