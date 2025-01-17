from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import time

# 1) Set up Firefox options and enable debug logs
firefox_options = Options()
firefox_options.log.level = "trace"  # Enable detailed logging
service = Service()  # Automatically uses the geckodriver in PATH or specify its path

driver = webdriver.Firefox(service=service, options=firefox_options)

try:
    # 2) Navigate to the hosted page
    driver.get("http://tiago-158c:8080/?&wtd=soFKtIIPPIrHMBD4")

    # 3) Give the page a moment to load
    time.sleep(3)

    # 4) Set new values for each relevant input
    # Arm_1 input (id="onn0pdb")
    arm_1_field = driver.find_element(By.ID, "onn0pdb")
    arm_1_field.clear()
    arm_1_field.send_keys("0.50")

    # Arm_2 input (id="onn0pdp")
    arm_2_field = driver.find_element(By.ID, "onn0pdp")
    arm_2_field.clear()
    arm_2_field.send_keys("-1.45")

    # Arm_3 input (id="onn0pe3")
    arm_3_field = driver.find_element(By.ID, "onn0pe3")
    arm_3_field.clear()
    arm_3_field.send_keys("-0.58")

    # ... Add more inputs as needed ...

    # 5) If there is a "submit" or "apply" button, you would click it:
    # submit_button = driver.find_element(By.ID, "some_button_id")
    # submit_button.click()

    # 6) Wait a few seconds to observe or allow the page to process changes
    time.sleep(5)

finally:
    # 7) Close the browser
    driver.quit()
