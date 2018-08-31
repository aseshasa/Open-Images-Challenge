import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to use
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('~/models/research/object_detection','data', 'oid_bbox_trainable_label_map.pbtxt')

# Path to test images
PATH_TO_TEST_IMAGES_DIR = 'challenge2018'

# Path to file containing class descriptions
PATH_TO_CLASSES_FILE = 'challenge-2018-class-descriptions-500.csv'

# Path to output file
OUTPUT_FILE = 'frcnn.csv'

NUM_CLASSES = 600

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def _ReadImageList(list_path):
	"""Helper function to read image paths.

	Args:
	list_path: Path to list of images, one image path per line.

	Returns:
	image_paths: List of image paths.
	"""
	with tf.gfile.GFile(list_path, 'r') as f:
		image_paths = f.readlines()
		image_paths = [entry.rstrip() for entry in image_paths]
		return image_paths

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
			(im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:
			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
			  'num_detections', 'detection_boxes', 'detection_scores',
			  'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
										tensor_name)
			if 'detection_masks' in tensor_dict:
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = sess.run(tensor_dict,
								 feed_dict={image_tensor: np.expand_dims(image, 0)})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict[
						'detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict			

def run_inference_for_many_images(list_images_path, graph):
	_STATUS_CHECK_ITERATIONS = 100
	out = []
	tf.logging.set_verbosity(tf.logging.INFO)
	# Read list of images.
	tf.logging.info('Reading list of images...')
	image_paths = _ReadImageList(list_images_path)
	num_images = len(image_paths)
	tf.logging.info('done! Found %d images', num_images)
	with graph.as_default():
		filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
		reader = tf.WholeFileReader()
		_, value = reader.read(filename_queue)
		image_tf = tf.image.decode_jpeg(value, channels=3)
		image_tf.set_shape([None, None, 3])
		image_tf = tf.expand_dims(image_tf, 0)
		with tf.Session() as sess:
			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
			  'num_detections', 'detection_boxes', 'detection_scores',
			  'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
										tensor_name)
			
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			start = time.clock()
			for i in range(num_images):
				if i == 0:
					tf.logging.info('Starting to detect objects in images...')
				elif i % _STATUS_CHECK_ITERATIONS == 0:
					elapsed = (time.clock() - start)
					tf.logging.info('Processing image %d out of %d, last %d '
								  'images took %f seconds', i, num_images,
								  _STATUS_CHECK_ITERATIONS, elapsed)
					start = time.clock()
				im = sess.run(image_tf)
				if 'detection_masks' in tensor_dict:
					# The following processing is only for single image
					detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
					detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
					# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
					real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
					detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
					detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
					detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
						detection_masks, detection_boxes, im.shape[0], im.shape[1])
					detection_masks_reframed = tf.cast(
						tf.greater(detection_masks_reframed, 0.5), tf.uint8)
					# Follow the convention by adding back the batch dimension
					tensor_dict['detection_masks'] = tf.expand_dims(
						detection_masks_reframed, 0)
				image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

				# Run inference
				output_dict = sess.run(tensor_dict,
									 feed_dict={image_tensor: im})

				# all outputs are float32 numpy arrays, so convert types as appropriate
				output_dict['num_detections'] = int(output_dict['num_detections'][0])
				output_dict['detection_classes'] = output_dict[
							'detection_classes'][0].astype(np.uint16)
				output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
				output_dict['detection_scores'] = output_dict['detection_scores'][0]
				if 'detection_masks' in output_dict:
					output_dict['detection_masks'] = output_dict['detection_masks'][0]
				output_dict['detection_file'] = image_paths[i]
				out.append(output_dict)
			coord.request_stop()
			coord.join(threads)
	return out

cat = {}
for i in label_map.item:
    cat[i.ListFields()[1][1]] = i.ListFields()[0][1]

import pandas as pd

cl500 = list(pd.read_csv(PATH_TO_CLASSES_FILE, header=None)[0])
files = os.listdir(PATH_TO_TEST_IMAGES_DIR)
RESET = 1000
with open(OUTPUT_FILE, 'w') as f:
	f.write('ImageId,PredictionString\n')
for k in range(0, len(files), RESET):
	with open('list_images.txt', 'w') as f:
		[f.write(os.path.join(os.getcwd(), PATH_TO_TEST_IMAGES_DIR, file)+'\n') for file in files[k:k+RESET]]
	output_dict = run_inference_for_many_images('list_images.txt', detection_graph)
	for j in range(RESET):
		sc = output_dict[j]['detection_scores']
		cl = output_dict[j]['detection_classes']
		bx = output_dict[j]['detection_boxes']
		str1 = ['%s %f %f %f %f %f' % (cat[cl[i]], sc[i], bx[i][0], bx[i][1], bx[i][2], bx[i][3]) for i in xrange(len(sc)) if sc[i] > 0.002 and cl[i] != 0 and cat[cl[i]] in cl500]
		with open('frcnn.csv', 'a') as f:
			f.write(output_dict[j]['detection_file'].split('/')[-1].split('.')[0] + ',' + ' '.join(str1) + '\n')
	print("%f%% complete" % (float(k)/len(files)*100))