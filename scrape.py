from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import numpy as np
from PIL import Image
import io

opt = webdriver.ChromeOptions()
opt.add_experimental_option('excludeSwitches', ['enable-logging'])
# opt.add_argument("--headless")

PATH = os.getcwd()

CHROMEDRIVER = PATH+"chromedriver.exe"

driver = webdriver.Chrome(executable_path=CHROMEDRIVER, options=opt)
driver.get(
    "https://www.deviantart.com/search/deviations?q=furry&order=most-recent&cursor=MTQwYWI2MjA9MTI2JjU5MGFjYWQwPTMwMDA")

index = 7978
while True:
    try:
        img_links = driver.find_elements(
            By.XPATH, "//img[contains(@src, 'https://images-wixmp')]")

        for i in img_links:
            try:
                print(f"INFO:: Downloading {index}.jpg")
                link = i.get_attribute("src")
                content = requests.get(link).content

                img = Image.open(io.BytesIO(content))
                img = img.resize((300, 300))
                img = img.convert("RGB")

                img.save(PATH+f"images/{index}.jpg", "JPEG", optimize=True)

                index += 1
            except Exception as e:
                print("ERROR:: Incorrect File Type")
                print(e)
                continue

        print("INFO:: Next")
        next = driver.find_elements(By.LINK_TEXT, "Next")[0].get_attribute("href")
        driver.get(next)
    except IndexError:
        print("ERROR:: No Next Found")
        print(driver.current_url)
        continue

print(driver.current_url)
