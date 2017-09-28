'''
San Wong (hswong1@uci.edu)
2017-09-12
The folloeing code is modified from "create_image_lists" in "retrain.py" 
Purpose: For some use case that we woluld like to seperate the Testing data set from Training data set
Therefore, this function suppose to provide an option to manually define the Training and Testing data set
'''




# Testing - Test if this function works
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import re
import hashlib

from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]




def create_image_lists_manual(image_dir, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.
    
  Args:
    image_dir: String path to a folder containing subfolders of training images.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
 

  '''
  =======================================================================
     
                          Image_Dir format (V2)
 
  =======================================================================
  Expected image_dir format (Ver2)
  img_dir
       
        --Class A
          --Train
              --- Image001
              --- Image002 ..... etc
          --Test
              --- Image001
              --- Image002 ..... etc
        --Class B
          --Train
              --- Image001
              --- Image002 ....
          --Test
              --- Image001
              --- Image002 ..... etc 

    '''          


  # Create Sub directory for Training objects
  sub_dirs= [x[0] for x in walklevel(image_dir, level=1)]
  is_root_dir = True
  result= {}
  
  sub_dir_iter = 0
  print("Reset sub_dir_iter: ",sub_dir_iter)
  for sub_dir in sub_dirs:
    sub_dir_iter = sub_dir_iter + 1
    print("sub_dir_iter: " , sub_dir_iter)
    print("curr_sub_dir")
    print(sub_dir)
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG'] 
    file_list_train=[]
    file_list_test=[]

    dir_name = os.path.basename(sub_dir) #dir_name 0s gonna be used to be image label

    tf.logging.info("Looking for images in '" + dir_name + "'")

    # Get a list of files that match the given patterns (pattern: file_glob_train/test + extension)
    Training = "Training"
    Testing = "Testing"
    for extension in extensions:
      file_glob_train = os.path.join(image_dir, dir_name,Training, '*.' + extension)
      file_list_train.extend(gfile.Glob(file_glob_train))
      file_glob_test = os.path.join(image_dir, dir_name,Testing, '*.' + extension)
      file_list_test.extend(gfile.Glob(file_glob_test))

    

    # Initialize return result
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    print(label_name)
    training_images = []
    testing_images = []
    validation_images = []



    # FOR LOOP: TRAINING
    file_name_iter = 0
    print("[TRAIN] Reset file_name_iter: ",file_name_iter)
    for file_name in file_list_train:
      file_name_iter = file_name_iter + 1
      print("[TRAIN] file_name_iter: ",file_name_iter)
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      else:
        training_images.append(base_name)

    

    # FOR LOOP: TESTING
    file_name_iter = 0
    print("[TEST] Reset file_name_iter: ",file_name_iter)
    for file_name in file_list_test:
      file_name_iter = file_name_iter+1
      print("[TEST] file_name_iter: ", file_name_iter)
      base_name = os.path.basename(file_name)
      testing_images.append(base_name)



     # Generate result
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
        }










  return result


def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  print("[]")
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
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
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
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




def main():
  #image_dir_test = "/Users/san/SourceTree/4axisRoboticArm_Automated_System/DVS_Inceptionv3_Tensorflow_Retrain/image_dir_test"
  #image_dir_train = "/Users/san/SourceTree/4axisRoboticArm_Automated_System/DVS_Inceptionv3_Tensorflow_Retrain/image_dir_train"
  image_dir = "/Users/san/SourceTree/4axisRoboticArm_Automated_System/DVS_Inceptionv3_Tensorflow_Retrain/image_dir_example"
  result = {}
  validation_percentage = 0.3
  result= create_image_lists_manual(image_dir, validation_percentage)
  print('Result')
  print(result['finn the human'])
  print(result['fionna'])



  image_dir_new = "/Users/san/SourceTree/4axisRoboticArm_Automated_System/DVS_Inceptionv3_Tensorflow_Retrain/image_dir_example1"
  result1 = {}
  testing_percentage = 0.2
  result1 = create_image_lists(image_dir_new, testing_percentage, validation_percentage)
  print('Result1')
  print(result1['finn the human'])
  print(result1['fionna'])





if __name__ == '__main__':
  main()

   