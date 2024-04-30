import cv2
import numpy as np
import os
from mmseg.apis import init_model, inference_model, show_result_pyplot


class DeepGreenSpaceRecognizer:

    PRED_SEG_STR = "pred_sem_seg"

    def __init__(self, config_path, checkpoint_path):
        # config_path = "../configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
        # checkpoint_path = "../checkpoints/iter_1000.pth"

        self.model = init_model(config_path, checkpoint_path, device="cuda:0")
        self.test_image_directory = None
        self.test_image_names = None

    def image_generator(self, test_image_directory, test_image_names):
        self.test_image_directory = test_image_directory
        self.test_image_names = test_image_names
        i = 0
        while i < len(test_image_names):
            image = cv2.imread(os.path.join(test_image_directory, test_image_names[i]))
            i += 1
            yield image

    def infer_batch(self, images):
        # Inference on a list of images
        inference = inference_model(self.model, images)

        # Extract numpy array of the predicted segmentation map
        seg_maps = []
        for result in inference:
            for item in result.numpy().items():
                if item[0] == DeepGreenSpaceRecognizer.PRED_SEG_STR:
                    seg_gray = np.squeeze(item[1].data, axis=0).astype(np.float32)
                    seg_maps.append(seg_gray)
                    # self.show_seg_map(seg_gray)
        return seg_maps

    @staticmethod
    def show_seg_map(seg_map):
        ret, thresh1 = cv2.threshold(seg_map, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow('Segmentation Map', thresh1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
