from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

options = Options()
options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
driver = webdriver.Chrome()
driver.maximize_window()
driver.get('https://livingatlas.arcgis.com/wayback/#active=47568&ext=-111.81786,41.74544,-111.81531,41.74735')
# driver.find_element(By.LINK_TEXT, '2021-08-11').click()

print("Chrome Browser Invoked")
# driver.close()