import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pyscreenshot as imagegrab
from src.utility import util
from src.utility.point import Point
import numpy as np


class ImageryWaybackDriver:
    def __init__(self, web_driver: webdriver):
        self.webdriver = web_driver

    @staticmethod
    def make_url(lon: float, lat: float, release_num: int = 51423, scale: int = 18) -> str:
        """
        Make an url of the Wayback Imagery website.
        :param lon: float, longitude of the center of the map
        :param lat: float, latitude of the center of the map
        :param release_num: int, an integer encoding date of the map
        :param scale: int, the scale factor for displaying the map
        :return: str, the url of the map on Wayback Imagery website.
        """
        return (f"https://livingatlas.arcgis.com/wayback/#active="
                f"{release_num}&mapCenter={round(lon, 5)}%2C{round(lat, 5)}%2C{scale:d}")

    def load_url(self, url: str):
        # Load the target webpage
        self.webdriver.get(url)
        self.webdriver.fullscreen_window()

    def toggle_off_version_filter(self):
        # Wait until version filter button is clickable and toggle it off
        (WebDriverWait(self.webdriver, 10)
         .until(EC.element_to_be_clickable((By.XPATH, "//*[@icon='check-square']")))
         .click())

    def close_modal(self):
        # Wait until the modal window is loaded and click close
        (WebDriverWait(self.webdriver, 10)
         .until(EC.element_to_be_clickable((By.XPATH, "//*[@aria-label='close-modal']")))
         .click())

    def accept_cookies(self):
        # Wait until the button "Accept Cookies" appears and click
        (WebDriverWait(self.webdriver, 10)
         .until(EC.element_to_be_clickable((By.XPATH, '//*[@id="onetrust-accept-btn-handler"]')))
         .click())

    def get_release_dates(self) -> dict[str, int]:
        """
        Get the mapping between dates of the map and the release numbers.
        :return: dict[str, int], {dates of the map: the release numbers}.
        """
        # Wait until the list view options appears for parsing release dates of maps
        WebDriverWait(self.webdriver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "py-1")))

        # Find all html tags containing the info of release dates
        page_source = self.webdriver.page_source
        bs = BeautifulSoup(page_source, 'html.parser')
        list_cards = bs.findAll('div', {'class': 'py-1'})
        # print(f"Number of date items: {len(list_cards)}")

        # Build a hashmap of release date and data-release-num
        date_to_release_num = dict()
        for card in list_cards:
            date = card.find('a', {'class': 'margin-left-half link-light-gray cursor-pointer'})
            date_to_release_num[date.get_text()] = card.attrs['data-release-num']
        return date_to_release_num

    def take_screenshot(self, width: int, height: int, save_to_file: str = None):
        """
        Take screenshot from the Wayback Imagery website.
        :param width: int, width of the screenshot
        :param height: int, height of the screenshot
        :param save_to_file: str, the path to save the screenshot. Default is None.
        :return: numpy array, the screenshot.
        """
        # Wait until the page is fully loaded
        WebDriverWait(self.webdriver, 10).until(lambda driver1: driver1.execute_script('return document.readyState') == 'complete')

        screen_w, screen_h = util.get_screen_resolution()
        nav_width = 350
        browser_header_size = 0  # Equal to 0 when full screen
        center = Point(screen_w // 2 + nav_width, screen_h // 2 + browser_header_size)
        shift = Point(width // 2, height // 2)  # (width / 2, height / 2) of the selected region to take a screenshot

        top_left = Point(center.x - shift.x, center.y - shift.y)
        bottom_right = Point(center.x + shift.x, center.y + shift.y)

        # Take a screenshot of the region of interest
        im = imagegrab.grab(bbox=(top_left.x, top_left.y, bottom_right.x, bottom_right.y))
        # im.show()
        if save_to_file is not None:
            im.save(save_to_file)

        return np.array(im)

    def close(self):
        self.webdriver.close()


def generate_train_data():
    top_left = (41.763356, -111.860899)
    bottom_right = (41.701922, -111.801124)
    xy_min = [min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])]
    xy_max = [max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])]
    n_samples = 100
    location_data = np.random.uniform(low=xy_min, high=xy_max, size=(n_samples, 2))
    # print(location_data)

    # Specify Chrome driver path
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

    # Start Chrome
    # driver = webdriver.Chrome(options=options)

    base_url = "https://livingatlas.arcgis.com/wayback/#active={:d}&mapCenter={:.6f}%2C{:.6f}%2C{:d}"
    release_num = 47963
    scale_factor = 18
    width = 512
    height = 1024

    path = f".\\image\\train_data\\RN{release_num}_C{scale_factor}_{width}X{height}_Num{n_samples}\\"
    if not os.path.exists(path):
        os.makedirs(path)

    base_filename = "green_space_"
    suffix = ".png"

    for i in range(location_data.shape[0]):
        driver = webdriver.Chrome(options=options)
        wayback = ImageryWaybackDriver(driver)
        url = base_url.format(release_num, location_data[i, 1], location_data[i, 0], scale_factor)
        wayback.load_url(url)
        wayback.toggle_off_version_filter()
        wayback.accept_cookies()
        filename = base_filename + str(i).zfill(len(str(n_samples))) + suffix
        save_to_file = os.path.join(path, filename)
        wayback.take_screenshot(width, height, save_to_file)
    print("Done!")


if __name__ == "__main__":
    # Specify Chrome driver path
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

    # Start Chrome
    driver = webdriver.Chrome(options=options)

    # Load the target webpage
    # url = 'https://livingatlas.arcgis.com/wayback/#active=47963&mapCenter=-111.82515%2C41.74474%2C17'
    # path = "../results/test_screenshot_C17.png"
    # scale_factor = 18
    #
    wayback_driver = ImageryWaybackDriver(driver)
    # wayback_driver.load_url(url)
    # wayback_driver.toggle_off_version_filter()
    # wayback_driver.accept_cookies()
    # release_dates = wayback_driver.get_release_dates()
    # wayback_driver.take_screenshot(512, 1024, path)
    # for key, value in release_dates.items():
    #     print(f"Date: {key}; Release num: {value}")

    url = wayback_driver.make_url(-111.876445354607, 41.75036406546104)
    wayback_driver.load_url(url)
    wayback_driver.toggle_off_version_filter()
    wayback_driver.accept_cookies()
    # release_dates = wayback_driver.get_release_dates()
    img = wayback_driver.take_screenshot(1510, 1510, "../image/test_screenshot.png")

    # print(type(img))
    # print(img.shape)
