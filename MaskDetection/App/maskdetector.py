import json
import platform
from typing import List, NamedTuple
import cv2
import numpy as np
from tflite_support import metadata
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

class Box(NamedTuple):
  left: float
  top: float
  right: float
  bottom: float


class Info(NamedTuple):
  label: str
  score: float
  index: int


class Detection(NamedTuple):
  bounding_box: Box
  categories: List[Info]

class MaskDetector:
  _OUTPUT_LOCATION_NAME = 'location'
  _OUTPUT_CATEGORY_NAME = 'category'
  _OUTPUT_SCORE_NAME = 'score'
  _OUTPUT_NUMBER_NAME = 'number of detections'

  def __init__(self,model_path):


    displayer = metadata.MetadataDisplayer.with_model_file(model_path)


    model_metadata = json.loads(displayer.get_metadata_json())
    process_units = model_metadata['subgraph_metadata'][0][
        'input_tensor_metadata'][0]['process_units']

    mean = 127.5
    std = 127.5

    for option in process_units:
      if option['options_type'] == 'NormalizationOptions':
        mean = option['options']['mean'][0]
        std = option['options']['std'][0]
    self._mean = mean
    self._std = std

    file_name = displayer.get_packed_associated_file_list()[0]
    label_map_file = displayer.get_associated_file_buffer(file_name).decode()
    label_list = list(filter(len, label_map_file.splitlines()))
    self._label_list = label_list
    
    interpreter = Interpreter(model_path=model_path, num_threads=4)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    sorted_output_indices = sorted(
        [output['index'] for output in interpreter.get_output_details()])
    
    self._output_indices = {
        self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
        self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
        self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
        self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
    }

    self._input_size = input_detail['shape'][2], input_detail['shape'][1]
    self._is_quantized_input = input_detail['dtype'] == np.uint8
    self._interpreter = interpreter

  def detect(self, input_image):
    (h,w)=(320,240)

    input_tensor = self.preprocess(input_image)

    self.set_input_tensor(input_tensor)
    self._interpreter.invoke()

    boxes = self.get_output_tensor(self._OUTPUT_LOCATION_NAME)
    classes = self.get_output_tensor(self._OUTPUT_CATEGORY_NAME)
    scores = self.get_output_tensor(self._OUTPUT_SCORE_NAME)
    count = int(self.get_output_tensor(self._OUTPUT_NUMBER_NAME))

    data = (boxes, classes, scores, count, h,w)

    return self.postprocess(data)



  def preprocess(self, input_image):
    input_tensor = cv2.resize(input_image, self._input_size)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor



  def set_input_tensor(self, image):
    tensor_index = self._interpreter.get_input_details()[0]['index']
    input_tensor = self._interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image



  def get_output_tensor(self, name):
    output_index = self._output_indices[name]
    tensor = np.squeeze(self._interpreter.get_tensor(output_index))
    return tensor



  def postprocess(self, data):
    boxes,classes,scores,count,image_width,image_height=data
    results = []
    threshold=0.4

    for i in range(count):
      if scores[i] >= threshold:

        y_min, x_min, y_max, x_max = boxes[i]

        bounding_box = Box(
            top=int(y_min * image_height),
            left=int(x_min * image_width),
            bottom=int(y_max * image_height),
            right=int(x_max * image_width))
        class_id = int(classes[i])

        category = Info(
            score=scores[i],
            label=self._label_list[class_id],
            index=class_id)
        
        result = Detection(bounding_box=bounding_box, categories=[category])
        results.append(result)

    sorted_results = sorted(
        results,
        key=lambda detection: detection.categories[0].score,
        reverse=True)

    filtered_results = sorted_results

    return filtered_results
