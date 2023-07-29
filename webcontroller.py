from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import shutil
from screenshotone import Client, TakeOptions

options = Options()
options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
driver = webdriver.Chrome()
driver.maximize_window()
url = 'https://livingatlas.arcgis.com/wayback/#active=47568&ext=-111.81786,41.74544,-111.81531,41.74735'
driver.get(url)
print("Chrome Browser Invoked")

element = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
    (By.XPATH, "//div[starts-with(@class, 'list-card trailer-half')]")))

# dates = driver.find_elements(By.XPATH, "//div[@class='list-card trailer-half']")
page_source = driver.page_source
bs = BeautifulSoup(page_source, 'html.parser')
# print(bs)
list_cards = bs.findAll('div', {'class': re.compile('list-card trailer-half+')})
print(f"Number of date items: {len(list_cards)}")

# Extract map release d
date_to_release_num = dict()
for card in list_cards:
    date = card.find('a', {'class': 'margin-left-half link-light-gray cursor-pointer'})
    date_to_release_num[date.get_text()] = card.attrs['data-release-num']

for key, value in date_to_release_num.items():
    print(f"Date: {key}; Release num: {value}")

# access_key = "Tw2yVHATZ8Bj5g"
# secret_key = "oo1iz_2kDB0fuQ"
# client = Client(access_key, secret_key)
#
# options = (TakeOptions.url(url)
#            .format("png")
#            .viewport_width(1920)
#            .viewport_height(1080)
#            .block_cookie_banners(True)
#            .block_chats(True))
#
# screenshot = client.take(options)
#
# with open("world_imagery_shot.png", "wb") as shot_file:
#     shutil.copyfileobj(screenshot, shot_file)

element = (WebDriverWait(driver, 10)
           .until(lambda driver1: driver1.execute_script('return document.readyState') == 'complete'))

driver.save_screenshot("world_imagery_shot.png")
# driver.close()

