import numpy as np
from imagery_wayback_driver import ImageryWaybackDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os


def generate_train_data():
    top_left = (41.763356, -111.860899)
    bottom_right = (41.701922, -111.801124)
    xy_min = [min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])]
    xy_max = [max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])]
    n_samples = 1000
    location_data = np.random.uniform(low=xy_min, high=xy_max, size=(n_samples, 2))
    # print(location_data)

    # Specify Chrome driver path
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

    # Start Chrome
    # driver = webdriver.Chrome(options=options)

    base_url = "https://livingatlas.arcgis.com/wayback/#active={:d}&mapCenter={:.6f}%2C{:.6f}%2C19"
    release_num = 47963

    path = ".\\image\\train_data\\C19_256X256\\"
    if not os.path.exists(path):
        os.makedirs(path)

    base_filename = "green_space_"
    suffix = ".png"

    for i in range(location_data.shape[0]):
        driver = webdriver.Chrome(options=options)
        wayback = ImageryWaybackDriver(driver)
        url = base_url.format(release_num, location_data[i, 1], location_data[i, 0])
        wayback.load_url(url)
        wayback.toggle_off_version_filter()
        wayback.accept_cookies()
        filename = base_filename + str(i).zfill(len(str(n_samples))) + suffix
        save_to_file = os.path.join(path, filename)
        wayback.take_screenshot(256, 256, save_to_file)
    print("Done!")


if __name__ == "__main__":
    generate_train_data()