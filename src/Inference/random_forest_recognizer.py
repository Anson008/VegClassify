import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
from utility import util
from pathlib import Path

class RandomForestRecognizer:
    def __init__(self, n_estimators=100, max_depth=20, n_jobs=-1):
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__n_jobs = n_jobs
        self.__train_index = set()
        self.__test_index = set()
        self.__x = None
        self.__y = None

    @staticmethod
    def extract_pixel_features(image: np.ndarray):
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        features = np.stack([
            image_rgb[:,:,0],
            image_rgb[:,:,1],
            image_rgb[:,:,2],
            x,
            y
        ], axis=-1)
        return features

    def build_train_index(self, split_file_path: str):
        if not os.path.exists(split_file_path):
            raise OSError

        with open(split_file_path, 'r') as file:
            for line in file:
                self.__train_index.add(line.strip()[-3:])

    def build_test_index(self, split_file_path: str):
        if not os.path.exists(split_file_path):
            raise OSError

        with open(split_file_path, 'r') as file:
            for line in file:
                self.__test_index.add(line.strip()[-3:])

    def build_feature(self, image_path: str, output_path: str):
        if not os.path.exists(image_path):
            raise OSError
        total = 0
        x_list = []
        with os.scandir(image_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".jpg") and self.__is_in_train_index(entry.name):
                    print(f">>> Building feature for {entry.name}")
                    image = cv2.imread(entry.path)
                    feats = RandomForestRecognizer.extract_pixel_features(image)
                    _, _, feature = feats.shape
                    x_list.append(feats.reshape(-1, feature))
                    total += 1

        self.__x = np.vstack(x_list)
        np.save(output_path, self.__x)
        print(f"Shape of feature: {self.__x.shape}")

    def build_label(self, label_path: str, output_path: str):
        if not os.path.exists(label_path):
            raise OSError

        y_list = []
        with (os.scandir(label_path) as entries):
            for entry in entries:
                if (entry.is_file() and
                    entry.name.endswith(".png") and
                    self.__is_in_index_set(entry.name, self.__train_index)):
                    print(f">>> Building label for {entry.name}")
                    label = cv2.imread(entry.path, flags=0)
                    y_list.append(label.reshape(-1))

        self.__y = np.hstack(y_list)
        np.save(output_path, self.__y)
        print(f"Shape of label: {self.__y.shape}")

    def __is_in_index_set(self, filename: str, index_set: set[str]):
        file_index = filename[-7: -4]
        return file_index in index_set

    def get_dataset_from_file(self, x_data_path: str, y_data_path: str):
        self.__x = np.load(x_data_path)
        self.__y = np.load(y_data_path)

    def fit(self, model_output_dir: Optional[str]):
        clf = RandomForestClassifier(n_estimators=self.__n_estimators,
                                     max_depth=self.__max_depth,
                                     n_jobs=self.__n_jobs)
        clf.fit(self.__x, self.__y)

        if model_output_dir:
            file_name = f"random_forest_estimators{self.__n_estimators}_depth{self.__max_depth}.joblib"
            full_path = os.path.join(model_output_dir, file_name)
            joblib.dump(clf, full_path)

    def predict(self, model, img: np.ndarray):
        feature = self.extract_pixel_features(img)
        height, width, feats = feature.shape

        x = feature.reshape(-1, feats)
        y = model.predict(x)

        mask = y.reshape(height, width)
        return mask

    def batch_predict(self, model, img_dir: str, output_dir: str):
        util.create_directory(output_dir)
        util.remove_all_files(output_dir)
        with os.scandir(img_dir) as entries:
            for entry in entries:
                if (entry.is_file() and
                    entry.name.endswith(".jpg") and
                    self.__is_in_index_set(entry.name, self.__test_index)):
                    img = cv2.imread(entry.path)
                    mask = self.predict(model, img)
                    mask[mask != 0] = 255
                    output_filename = f"{Path(entry.path).stem}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, mask)



if __name__ == "__main__":
    # split_file_path = "D:\\DeepGreenSpace_Train_Data\\Naip_National_Labeled_200_voc\\splits\\train.txt"
    # image_path = "D:\\DeepGreenSpace_Train_Data\\Naip_National_Labeled_200_voc\\images"
    # label_path = "D:\\DeepGreenSpace_Train_Data\\Naip_National_Labeled_200_voc\\labels"
    # feature_output_path = "D:\\RandomForestNDVI\\feature.npy"
    # label_output_path = "D:\\RandomForestNDVI\\label.npy"
    # model_output_dir = "D:\\RandomForestNDVI\\model_zoo"

    # Build training data
    # rf_classifier = RandomForestRecognizer()
    # rf_classifier.build_train_index(split_file_path)
    # rf_classifier.build_feature(image_path, feature_output_path)
    # rf_classifier.build_label(label_path, label_output_path)

    # Train
    # rf_classifier = RandomForestRecognizer()
    # rf_classifier.get_dataset_from_file(feature_output_path, label_output_path)
    # print("Training...")
    # rf_classifier.fit(model_output_dir)
    # print("Done!")

    # Predict
    # test_index_file_path = "D:\\DeepGreenSpace_Train_Data\\Naip_National_Labeled_200_voc\\splits\\test.txt"
    # test_image_path = "D:\\DeepGreenSpace_Train_Data\\Naip_National_Labeled_200_voc\\images"
    # model_path = "D:\\RandomForestNDVI\\model_zoo\\random_forest_estimators100_depth20.joblib"
    # test_output_path = "D:\\RandomForestNDVI\\prediction\\rf_estimators100_depth20"
    #
    # model = joblib.load(model_path)
    # test_img = cv2.imread(test_image_path)
    # rf_classifier = RandomForestRecognizer()
    # rf_classifier.build_test_index(test_index_file_path)
    #
    # mask_pred = rf_classifier.predict(model, test_img)
    # cv2.imshow("output", mask_pred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # np.save(test_output_path, mask_pred)

    # Batch predict
    test_index_file_path = "D:\\DeepGreenSpace_Train_Data\\Naip_National_Labeled_200_voc\\splits\\test.txt"
    test_image_path = "D:\\DeepGreenSpace_Train_Data\\Naip_National_Labeled_200_voc\\images"
    model_path = "D:\\RandomForestNDVI\\model_zoo\\random_forest_estimators100_depth20.joblib"
    test_output_path = "D:\\RandomForestNDVI\\prediction\\rf_estimators100_depth20"

    model = joblib.load(model_path)
    rf_classifier = RandomForestRecognizer()
    rf_classifier.build_test_index(test_index_file_path)
    rf_classifier.batch_predict(model, test_image_path, test_output_path)
