import cv2
import os
import shutil
from pathlib import Path
import utility.util


def find_original_naip(target_image_path: Path, original_naip_dir: Path, output_path: Path):
    target_image = cv2.imread(str(target_image_path), cv2.IMREAD_GRAYSCALE)

    with os.scandir(original_naip_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".png"):
                original_naip_image = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)
                similarity = cv2.matchTemplate(original_naip_image, target_image, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, _, _ = cv2.minMaxLoc(similarity)
                if max_val >= 0.95:
                    with open(output_path, "a") as f:
                        f.write(f"{target_image_path.name}: {entry.name}\n")
                        return True
    return False

def find_all_original_naip(jpg_image_dir: Path, original_naip_dir_root: Path, output_path: Path):
    with os.scandir(jpg_image_dir) as jpg_entries:
        for jpg_entry in jpg_entries:
            if jpg_entry.is_file() and jpg_entry.name.endswith(".jpg"):
                with os.scandir(original_naip_dir_root) as entries:
                    for entry in entries:
                        if entry.is_dir() and entry.name.endswith("_split"):
                            if find_original_naip(Path(jpg_entry.path), Path(entry.path), output_path):
                                break

def copy_test_data(filename_dir: Path, destination_dir: Path, source_dir: Path):
    utility.util.create_directory(destination_dir)
    with os.scandir(filename_dir) as filename_entries:
        for filename_entry in filename_entries:
            if filename_entry.is_file() and filename_entry.name.endswith(".tif"):
                dir_name = filename_entry.name[:28]
                with os.scandir(source_dir) as source_dir_entries:
                    for source_dir_entry in source_dir_entries:
                        if source_dir_entry.is_dir() and source_dir_entry.name[:-6] == dir_name:
                            copy_source = os.path.join(source_dir_entry, filename_entry.name)
                            shutil.copy(copy_source, destination_dir)
