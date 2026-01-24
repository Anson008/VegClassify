import json
import os
import shutil
import pandas as pd
from pathlib import Path
from utility.confusion_matrix import ConfusionMatrix
from utility.image_editor import ImageEditor
from utility.mask_factory import FullMaskCreator

metrics = ("accuracy", "precision", "recall", "f1score", "kappa")

def read_file_names(name_list_path : Path):
    jpg_to_naip = dict()
    with open(name_list_path, 'r') as f:
        for line in f:
            name_split = line.split(":")
            jpg_to_naip[name_split[0].strip()] = name_split[1].strip()
    return jpg_to_naip

def get_accuracy(filename_list_path: Path, annotation_dir: Path, segmentation_dir: Path, output_path: Path):
    jpg_to_naip = read_file_names(filename_list_path)
    statistics = dict()
    statistics["filename"] = []
    confusion_matrix = ConfusionMatrix()
    count = 0
    for key, val in jpg_to_naip.items():
        annotation_path = annotation_dir.joinpath(key)
        annotation_path = str(annotation_path)[:-4] + ".png"
        segmentation_path = segmentation_dir.joinpath(val)
        segmentation_path = str(segmentation_path)[:-4] + ".tif_vegetation_mask.tif"
        annotation_mask = ImageEditor.load_png(annotation_path)
        segmentation_mask = ImageEditor.load_raster(segmentation_path)
        confusion_matrix.compute_on_single_sample(annotation_mask, segmentation_mask)
        cm = confusion_matrix.get_confusion_matrix()
        statistics["filename"].append(val[:-4])
        for metric in metrics:
            if metric not in statistics.keys():
                statistics[metric] = [cm[metric]]
            else:
                statistics[metric].append(cm[metric])
        count += 1
        confusion_matrix.reset()
    df = pd.DataFrame(data=statistics)
    df.to_csv(output_path, index=False)
    # for key, val in statistics.items():
    #     statistics[key] = val / count
    # with open(output_path, "w") as outfile:
    #     outfile.write(json.dumps(statistics, indent=4))

def move_deep_model_results(filename_list_path: Path, source: Path, destination: Path):
    filenames = read_file_names(filename_list_path)
    for val in filenames.values():
        filename = val[:-4]
        source_path = source.joinpath(filename)
        with os.scandir(source_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".png"):
                    destination_path = destination.joinpath(val)
                    shutil.copy(entry.path, destination_path)

def get_accuracy_for_learning_models(filename_list_path: Path, annotation_dir: Path, segmentation_dir: Path, output_path: Path):
    jpg_to_naip = read_file_names(filename_list_path)
    statistics = dict()
    statistics["filename"] = []
    confusion_matrix = ConfusionMatrix()
    for key, val in jpg_to_naip.items():
        annotation_path = annotation_dir.joinpath(key)
        annotation_path = str(annotation_path)[:-4] + ".png"
        segmentation_path = segmentation_dir.joinpath(val)
        segmentation_path = str(segmentation_path)
        annotation_mask = ImageEditor.load_png(annotation_path)
        segmentation_mask = ImageEditor.load_png(segmentation_path)
        confusion_matrix.compute_on_single_sample(annotation_mask, segmentation_mask)
        cm = confusion_matrix.get_confusion_matrix()
        statistics["filename"].append(val[:-4])
        for metric in metrics:
            if metric not in statistics.keys():
                statistics[metric] = [cm[metric]]
            else:
                statistics[metric].append(cm[metric])
        confusion_matrix.reset()
    df = pd.DataFrame(data=statistics)
    df.to_csv(output_path, index=False)
    # with open(output_path, "w") as outfile:
    #     outfile.write(json.dumps(cm, indent=4))

def copy_test_annotation(filename_path: Path, destination_dir: Path, source_dir: Path):
    jpg_to_tif = read_file_names(filename_path)
    for key, val in jpg_to_tif.items():
        source_path = source_dir.joinpath(key[:-4] + ".png")
        destination_path = destination_dir.joinpath(val)
        shutil.copy(source_path, destination_path)


if __name__ == "__main__":
    # SmartNDVI
    # name_map_path = Path("D:\\Accuracy_Assessment\\JPGTONAIP_TestData.txt")
    # annotation_dir = Path("D:\\Accuracy_Assessment\\Manually_Annotated_Data")
    # segmentation_dir = Path("D:\\Accuracy_Assessment\\SmartNDVI_Test_Data_Run")
    # output_path = Path("D:\\Accuracy_Assessment\\ndvi_accuracy_stats_output.csv")
    # get_accuracy(name_map_path, annotation_dir, segmentation_dir, output_path)

    # filename_path = Path("D:\\Accuracy_Assessment\\JPGTONAIP_TestData.txt")
    # deep_results_path = Path("D:\\naip_playground2\\cache\\ground_truth_mask")
    # destination_path = Path("D:\\Accuracy_Assessment\\Deep_Model_Results")
    # move_deep_model_results(filename_path, deep_results_path, destination_path)

    # Deep learning
    # name_map_path = Path("D:\\Accuracy_Assessment\\JPGTONAIP_TestData.txt")
    # annotation_dir = Path("D:\\Accuracy_Assessment\\Manually_Annotated_Data")
    # deep_segmentation_dir = Path("D:\\Accuracy_Assessment\\Deep_Model_Results")
    # output_path = Path("D:\\Accuracy_Assessment\\deep_learning_accuracy_output_1.csv")
    # get_accuracy_for_deep_results(name_map_path, annotation_dir, deep_segmentation_dir, output_path)

    # Rename manually annotated data
    # filename_path = Path("D:\\Accuracy_Assessment\\JPGTONAIP_TestData.txt")
    # destination_dir = Path("D:\\Accuracy_Assessment\\Manually_Annotated_Test_Data")
    # source_dir = Path("D:\\Accuracy_Assessment\\Manually_Annotated_Data")
    # copy_test_annotation(filename_path, destination_dir, source_dir)

    # Rename Random Forest prediction
    # filename_path = Path("D:\\Accuracy_Assessment\\JPGTONAIP_TestData.txt")
    # destination_dir = Path("D:\\Accuracy_Assessment\\RandomForest_Prediction_Results_estimators400")
    # source_dir = Path("D:\\RandomForestNDVI\\prediction\\rf_estimators400_depth20")
    # copy_test_annotation(filename_path, destination_dir, source_dir)

    # Random Forest
    name_map_path = Path("D:\\Accuracy_Assessment\\JPGTONAIP_TestData.txt")
    annotation_dir = Path("D:\\Accuracy_Assessment\\Manually_Annotated_Data")
    rf_classification_dir = Path("D:\\Accuracy_Assessment\\RandomForest_Prediction_Results_estimators400")
    output_path = Path("D:\\Accuracy_Assessment\\random_forest_accuracy_output_estimators400.csv")
    get_accuracy_for_learning_models(name_map_path, annotation_dir, rf_classification_dir, output_path)
