import time
import pyautogui as pag

def click(delay: float):
	pag.mouseDown()
	time.sleep(delay)
	pag.mouseUp()