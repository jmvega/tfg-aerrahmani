import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from threading import Thread
import imutils
from imutils.video import FPS
import time

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
CONFIG_PATH = 'models'+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
PRETRAINED_MODEL_PATH='pre-trained-models'
APIMODEL_PATH = 'models'
CHECKPOINT_PATH = 'models/my_ssd_mobnet/'


# converter = tf.lite.TFLiteConverter.from_saved_model("models/person_detect/tfliteexport/saved_model")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quant_model = converter.convert()


# config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
# with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
#     proto_str = f.read()                                                                                                                                                                                                                                          
#     text_format.Merge(proto_str, pipeline_config) 

# pipeline_config.model.ssd.num_classes = 1
# pipeline_config.train_config.batch_size = 4
# pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
# pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
# pipeline_config.train_input_reader.label_map_path= 'annotation' + '/label_map.pbtxt'
# pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ['annotation' + '/train.record']
# pipeline_config.eval_input_reader[0].label_map_path = 'annotation' + '/label_map.pbtxt'
# pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ['annotation' + '/test.record']

# config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
# with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
#     f.write(config_text) 

# print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=5000""".format(APIMODEL_PATH, APIMODEL_PATH,CUSTOM_MODEL_NAME,APIMODEL_PATH,CUSTOM_MODEL_NAME))

import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

import cv2 
import numpy as np

category_index = label_map_util.create_category_index_from_labelmap('annotation'+'/label_map.pbtxt')

# cap.release()

# Setup capture
# TFLITE_PATH='models/person_detect/tfliteexport'
# TFLITE_SCRIPT = os.path.join(APIMODEL_PATH, 'research', 'object_detection', 'export_tflite_graph_tf2.py ')
# command = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT ,CONFIG_PATH, CHECKPOINT_PATH, TFLITE_PATH)
# # print(command)
# FROZEN_TFLITE_PATH = os.path.join(TFLITE_PATH, 'saved_model')
# TFLITE_MODEL = os.path.join(TFLITE_PATH, 'saved_model', 'detect.tflite')

# command = "tflite_convert \
# --saved_model_dir={} \
# --output_file={} \
# --input_shapes=1,300,300,3 \
# --input_arrays=normalized_input_image_tensor \
# --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
# --inference_type=FLOAT \
# --allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL, )
# print(command)





class WebcamVideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3,320)
        self.stream.set(4,240)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):

        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True


vs = WebcamVideoStream(0).start()
# vs=cv2.VideoCapture(0)
# vs.set(3,320)
# vs.set(4,320)
fps = FPS().start()
counter, fps = 0, 0
fps_avg_frame_count = 10
start_time = time.time()

while True: 
    frame = vs.read()

    frame = imutils.resize(frame, width=320)

    image_np = np.array(frame)

    # image_np=image_np[100:200,50:200]

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=.5,
                agnostic_mode=True)

    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
    
    fps_text = 'FPS = {:.1f}'.format(fps/10)

    # print(fps_text)

    cv2.putText(image_np_with_detections, fps_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('object detection',  image_np_with_detections)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        # cap.release()
        vs.stop()
        break
