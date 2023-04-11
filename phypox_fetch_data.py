from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys, ActionChains
import time
import requests
import geocoder
from selenium.webdriver.support.ui import Select

options=ChromeOptions()
options.add_argument("--disable-extensions")
options.add_argument("--incognito")
# options.add_argument("--proxy-server='direct://'")
# options.add_argument("--proxy-bypass-list=*")
options.add_argument("--start-maximized")
#options.add_argument('--headless')    
options.add_argument('--disable-gpu')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--no-sandbox')
options.add_argument('--ignore-certificate-errors')
user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
options.add_argument(f'user-agent={user_agent}')

options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=options, service=ChromeService(executable_path=ChromeDriverManager().install()))
driver.get('http://192.168.2.14/')

# Find the unordered list by ID
ulist = driver.find_element(By.ID, 'viewSelector')
# Find the first list item under the unordered list by XPath
li = ulist.find_element(By.XPATH, './/li[2]')

# Change the class of the list item
li.click()

while True:
    acceleration =  WebDriverWait(driver, timeout=10).until(lambda d: d.find_element(By.CLASS_NAME, 'valueNumber'))
    print(acceleration.text)
                                                                                     
