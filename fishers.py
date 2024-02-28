import time

import pytesseract
import pyautogui as pag
import numpy as np
import cv2

from utils import click

pytesseract.pytesseract.tesseract_cmd = r"D:\"

class SimpleFisher:
	
	def __init__(self):
		self.x_offset = -200
		self.y_offset = -60
		self.width = 400
		self.height = 120
		self.lower_rod = [0, 50, 50]
		self.upper_rod = [10, 255, 255]
		
		self.rod_icon = "testrodicon.png"
	
	def start(self):
		while True:
			cur_x, cur_y = pag.position()
			screen = pag.screenshot("temp.png", region=(cur_x + self.x_offset, cur_y + self.y_offset, self.width, self.height))
			located = pag.locate(self.rod_icon, "temp.png", confidence = 0.5)
			
			hsv = cv2.cvtColor(cv2.imread("temp.png"), cv2.COLOR_BGR2HSV)
			lower_rod = np.array(self.lower_rod)
			upper_rod = np.array(self.upper_rod)
			mask = cv2.inRange(hsv, lower_rod, upper_rod)
			
			if not np.sum(mask) > 0:
				print("Rod is not detected! Drop!")
				click(0.02)
				time.sleep(1)
			elif located != None:
				print("Vibrations Detected! Catch!")
				click(0.02)
				time.sleep(1.5)

class SonarFisher(SimpleFisher):
	
	def __init__(self):
		SimpleFisher.__init__(self)
		self.lang = "rus"
	
	def recognize_text(self, image):
		return self.recognize_text2(image);
	
	def recognize_text2(self, image):
		original = image.copy()
		mask = np.zeros(image.shape, dtype=np.uint8) 
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

		# Find contours and filter using aspect ratio and area
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		for c in cnts:
			area = cv2.contourArea(c)
			x,y,w,h = cv2.boundingRect(c)
			ar = w / float(h)
			if area > 1000 and ar > .85 and ar < 1.2:
				cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
				cv2.rectangle(mask, (x, y), (x + w, y + h), (255,255,255), -1)
				ROI = original[y:y+h, x:x+w]

		# Bitwise-and to isolate characters 
		result = cv2.bitwise_and(original, mask)
		result[mask==0] = 255
		
		#mine
		cv2.imshow("mask", mask)
		cv2.imshow("gray", gray)
		cv2.imshow("thresh", thresh)
		cv2.imshow("result", result)
		cv2.waitKey()
		
		# OCR
		data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')
		return data
	
	def recognize_text1(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#blur = cv2.GaussianBlur(gray, (3,3), 0)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

		#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
		invert = 255 - thresh#||||||||||||||||

		data = pytesseract.image_to_string(invert, lang=self.lang, config="--psm 6")

		cv2.imshow("image", image)
		cv2.imshow('thresh', thresh)
		#cv2.imshow('opening', opening)
		cv2.imshow('invert', invert)
		cv2.waitKey()
		
		return data
	
	def start(self):
		#while True:
		cur_x, cur_y = pag.position()
		pag.screenshot("temp.png", region=(cur_x + self.x_offset, cur_y + self.y_offset, self.width, self.height))
		
		im = cv2.imread("tesseracttest.png")
		txt = self.recognize_text(im)
		print(txt)
			
			#if not self.detect_rod(self.get_mask("temp.png")) > 0:
				#click(0.02)
