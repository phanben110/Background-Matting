import math
import os

import cv2
import numpy as np
from modelplace_api import BaseModel, BBox, Device, Landmarks, Point, TaskType
from openvino.inference_engine import IECore, IENetwork

from .postprocessing import Postprocessor


def pad_img(img, pad_value, target_dims):
    h, w, _ = img.shape
    pads = []
    pads.append(int(math.floor((target_dims[0] - h) / 2.0)))
    pads.append(int(math.floor((target_dims[1] - w) / 2.0)))
    pads.append(int(target_dims[0] - h - pads[0]))
    pads.append(int(target_dims[1] - w - pads[1]))
    padded_img = cv2.copyMakeBorder(
        img, pads[0], pads[2], pads[1], pads[3], cv2.BORDER_CONSTANT, value=pad_value,
    )
    return padded_img, pads


class InferenceModel(BaseModel):
    def __init__(
        self,
        model_path: str = "",
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        **kwargs,
    ):
        model_path = (
            model_path
            if model_path != ""
            else os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoints")
        )
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.threshold = threshold
        self.class_names = {0: "background", 1: "palm"}

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)
            height, width, _ = img.shape
            if self.input_height / self.input_width < height / width:
                scale = self.input_height / height
            else:
                scale = self.input_width / width

            scaled_img = cv2.resize(
                img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
            )
            padded_img, pad = pad_img(
                scaled_img, (0, 0, 0), [self.input_height, self.input_width],
            )
            padded_img = padded_img / 128 - 1
            padded_img = padded_img.transpose((2, 0, 1))
            padded_img = padded_img[np.newaxis].astype(np.float32)
            preprocessed_data.append(padded_img)
            data_infos.append((scale, pad))

        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        if not len(predictions[0]):
            return [[]]
        postprocessed_result = []

        for result, input_info in zip(predictions[0], predictions[1]):
            scale, pads = input_info
            h, w = self.input_height, self.input_width
            original_h = int((h - (pads[0] + pads[2])) / scale)
            original_w = int((w - (pads[1] + pads[3])) / scale)
            image_predictions = []
            boxes, keypoints = self.postprocessor.decode_predictions(
                result, self.output_names,
            )
            for box, kps in zip(boxes, keypoints):
                if box[4] > self.threshold:
                    palm_box = BBox(
                        x1=int(np.clip((box[0] * w - pads[1]) / scale, 0, original_w)),
                        y1=int(np.clip((box[1] * h - pads[0]) / scale, 0, original_h)),
                        x2=int(np.clip((box[2] * w - pads[1]) / scale, 0, original_w)),
                        y2=int(np.clip((box[3] * h - pads[0]) / scale, 0, original_h)),
                        score=float(box[4]),
                        class_name=self.class_names[1],
                    )
                    image_predictions.append(
                        Landmarks(
                            bbox=palm_box,
                            keypoints=[
                                Point(
                                    x=int((keypoint[0] * w - pads[1]) / scale),
                                    y=int((keypoint[1] * h - pads[0]) / scale),
                                )
                                for keypoint in kps
                            ],
                        ),
                    )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def model_load(self, device=Device.cpu):
        self.task_type = TaskType.landmark_detection

        model_xml = os.path.join(self.model_path, "palm_detection_builtin.xml")
        model_bin = os.path.join(self.model_path, "palm_detection_builtin.bin")
        self.net = IECore().load_network(
            network=IENetwork(model=model_xml, weights=model_bin),
            device_name="CPU",
            num_requests=2,
        )
        self.input_name = next(iter(self.net.inputs))
        self.output_names = sorted(self.net.outputs)
        _, _, self.input_height, self.input_width = self.net.inputs[
            self.input_name
        ].shape
        self.postprocessor = Postprocessor(
            os.path.join(os.path.dirname(__file__), "ssd_anchors.csv"),
        )

    def forward(self, data):
        data[0] = [
            self.net.infer(inputs={self.input_name: sample}) for sample in data[0]
        ]
        return data

    def process_sample(self, image):
        data = self.preprocess([image])
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]
