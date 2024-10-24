import numpy as np
import cv2
import os
from morphology.connected_components import ConnectedComponentsProcessor as ccp
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import pickle


class ColorDetector:
    def __init__(self, path=None):
        if path:
            self._img = cv2.imread(path)
        else:
            self._img = None

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, path):
        self._img = cv2.imread(path)

    def apply_hist_equalization(self, channel=2):
        hsv_img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
        hsv_img[:, :, channel] = cv2.equalizeHist(hsv_img[:, :, channel])
        return hsv_img, cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    def detect_color_hsv(self, lower_hue=30, upper_hue=90):
        lower = np.array([lower_hue, 30, 95])
        upper = np.array([upper_hue, 255, 255])
        hsv_img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_img, lower, upper)

    def detect_color_em(self, train_img, n_class=2, mode=None, save_to_file="./models/gaussian_mixture/em_model.pickle"):
        train_data = cv2.cvtColor(train_img, cv2.COLOR_BGR2HSV)
        train_data = self._standardize(np.copy(train_data))
        x_train = train_data.reshape(train_data.shape[0] * train_data.shape[1], -1)
        if mode == "grid-search":
            param_grid = {"n_components": [4, 5, 6]}
            gm = GaussianMixture(n_init=15, max_iter=300, tol=1e-4, init_params="random_from_data", verbose=1)
            gm_model = GridSearchCV(gm, param_grid=param_grid, scoring=self._bic_score)
            gm_model.fit(x_train)
            print(gm_model.best_params_)
        else:
            gm_model = GaussianMixture(n_init=15,
                                       init_params="random_from_data",
                                       max_iter=300,
                                       n_components=n_class,
                                       tol=1e-4,
                                       verbose=1)
            gm_model.fit(x_train)
        pickle.dump(gm_model, open(save_to_file, "wb"))

    def predict_em(self, model_path):
        model = pickle.load(open(model_path, "rb"))
        test_data = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
        test_data = test_data.reshape(test_data.shape[0] * test_data.shape[1], -1)
        y_pred = model.predict(test_data)
        return y_pred.reshape(self._img.shape[0], self._img.shape[1])

    @staticmethod
    def _standardize(x):
        return (x - x.mean(axis=(0, 1, 2), keepdims=True)) / x.std(axis=(0, 1, 2), keepdims=True)

    @staticmethod
    def _bic_score(estimator, x):
        return -estimator.bic(x)


def make_em_train_data(dest_filename, source_dir):
    images = []
    for filename in os.listdir(source_dir):
        if filename.startswith("world_imagery") and filename.endswith(".png"):
            f = os.path.join(source_dir, filename)
            if os.path.isfile(f):
                images.append(cv2.imread(f))

    image = images[0]
    for i in range(1, len(images)):
        image = cv2.hconcat([image, images[i]])

    cv2.imwrite(dest_filename, image)


if __name__ == "__main__":
    detector = ColorDetector("../image/naip_rgb.png")
    bgr_img = detector.img

    # mask = detector.detect_color()
    # res = ccp.overlap_on_map(mask, bgr_img, "red")
    # cv2.imwrite('./results/color_detection_H30-90_S30-255_V95-255.png', res)

    # make_em_train_data("./image/em_train_img_01.png", "./results")

    file_path = "../../models/gaussian_mixture/gm_init15_randFromData_class6_tol1e-4.pickle"
    # train_img = cv2.imread("./image/em_train_img_colors.png")
    # em_mode = "grid-search"
    # detector.detect_color_em(train_img, n_class=6, save_to_file=file_path)

    seg_img = detector.predict_em(file_path)
    print(seg_img.shape)
    print(seg_img.min(), seg_img.max())
    seg_img = seg_img.astype("uint8")
    # plt.imshow(seg_img)
    # plt.show()
    for i in range(1, 4):
        temp = np.copy(seg_img)
        temp[temp == i] = 255
    # seg_img[seg_img == 0] = 255
    # seg_img[seg_img == 1] = 0
    # print(type(seg_img), seg_img.dtype, seg_img.shape, seg_img)
        res = ccp.overlap_on_map(temp, bgr_img, "red")
    cv2.imwrite("../results/em_hsv_trained_init15_class6_tol1e-4.png", res)
    # plt.imshow(seg_img)
    # plt.show()

