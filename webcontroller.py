from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import pyscreenshot as imagegrab
import time

# Specify Chrome driver path
options = Options()
options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

# Start Chrome
driver = webdriver.Chrome()
driver.maximize_window()

# Load the target webpage
url = 'https://livingatlas.arcgis.com/wayback/#active=47963&mapCenter=-111.821131%2C41.743871%2C19'
driver.get(url)
print("Chrome Browser Invoked")

# WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "flex items-center justify-center cursor-pointer my-1")))
# local_change_check_box = driver.find_element(By.CLASS_NAME, "flex items-center justify-center cursor-pointer my-1")
# if local_change_check_box.is_selected():
#     print("Selected. Toggle off.")
#     local_change_check_box.click()

# Wait until the button "Accept Cookies" appears and click
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="onetrust-accept-btn-handler"]'))).click()

# Wait until the list view options appears for parsing release dates of maps
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "py-1")))

# Find all html tags containing the info of release dates
page_source = driver.page_source
bs = BeautifulSoup(page_source, 'html.parser')
list_cards = bs.findAll('div', {'class': 'py-1'})
print(f"Number of date items: {len(list_cards)}")

# Build a hashmap of release date and data-release-num
date_to_release_num = dict()
for card in list_cards:
    date = card.find('a', {'class': 'margin-left-half link-light-gray cursor-pointer'})
    date_to_release_num[date.get_text()] = card.attrs['data-release-num']

for key, value in date_to_release_num.items():
    print(f"Date: {key}; Release num: {value}")

# Wait until the page is fully loaded
WebDriverWait(driver, 10).until(lambda driver1: driver1.execute_script('return document.readyState') == 'complete')

# Take a screenshot of the region of interest
im = imagegrab.grab(bbox=(700, 150, 3840, 2160))
im.show()
# im.save("./results/world_imagery_screenshot.png")

# driver.save_screenshot("world_imagery_shot.png")
# driver.close()

