import numpy as np
import tensorflow as tf
from PIL import Image

# colour map
label_colours = [(0, 0, 0)
                 # 0=background
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

#for LUNA 16
# label_colours = [(0,0,0)
#                 # 0=background
#                 ,(128,0,0),(0,128,0),(128,128,0),(0,0,128)]
                # 1=left lung, 2=right lung, 3=pipe, 4=nodule,

#for ILD
# label_colours = [(0,0,0)
#                 # 0=background
#                 ,(128,128,0),(0,0,128)]
#                 # 1= lungs, 2=nodules,
#for ImageNet2016 Scene-parsing
# label_colours = [(0,0,0), (251,209,244), (44,230,121), (156,40,149), (166,219,98), (35,229,138), (143,56,194), (144,223,70), (200,162,57), (120,225,199), (87,203,13), (185,1,136), (16,167,16), (29,249,241), (17,192,40), (199,44,241), (193,196,159), (241,172,78), (56,94,128), (231,166,116), (50,209,252), (217,56,227), (168,198,178), (77,179,188), (236,191,103), (248,138,151), (214,251,89), (208,204,187), (115,104,49), (29,202,113), (159,160,95), (78,188,13), (83,203,82), (8,234,116), (80,159,200), (124,194,2), (192,146,237), (64,3,73), (17,213,58), (106,54,105), (125,72,155), (202,36,231), (79,144,4), (118,185,128), (138,61,178), (23,182,182), (154,114,4), (201,0,83), (21,134,53), (194,77,237), (198,81,106), (37,222,181), (203,185,14), (134,140,113), (220,196,79), (64,26,68), (128,89,2), (199,228,65), (62,215,111), (124,148,166), (221,119,245), (68,57,158), (80,47,26), (143,59,56), (14,80,215), (212,132,31), (2,234,129), (134,179,44), (53,21,129), (80,176,236), (154,39,168), (221,44,139), (103,56,185), (224,138,83), (243,93,235), (80,158,63), (81,229,38), (116,215,38), (103,69,182), (66,81,5), (96,157,229), (164,49,170), (14,42,146), (164,67,44), (108,116,151), (144,8,144), (85,68,228), (16,236,72), (108,7,86), (172,27,94), (119,247,193), (155,240,152), (49,158,204), (23,193,204), (228,66,107), (69,36,163), (238,158,228), (202,226,35), (194,243,151), (192,56,76), (16,115,240), (61,190,185), (7,134,32), (192,87,171), (45,11,254), (179,183,31), (181,175,146), (13,187,133), (12,1,2), (63,199,190), (221,248,32), (183,221,51), (90,111,162), (82,0,6), (40,0,239), (252,81,54), (110,245,152), (0,187,93), (163,154,153), (134,66,99), (123,150,242), (38,144,137), (59,180,230), (144,212,16), (132,125,200), (26,3,35), (199,56,92), (83,223,224), (203,47,137), (74,74,251), (246,81,197), (168,130,178), (136,85,200), (186,147,103), (170,21,85), (104,52,182), (166,147,202), (103,119,71), (74,161,165), (14,9,83), (129,194,43), (7,100,55), (13,12,170), (30,21,22), (224,189,139), (40,77,25), (194,14,94), (178,8,231), (234,166,8), (248,25,7), (139,181,248)]
def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    # mask = np.expand_dims(mask, axis=-1)
    # try:
    #     mask = np.squeeze(mask, axis=4)
    # except:
    #     pass
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs
