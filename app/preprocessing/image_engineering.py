import cv2
import numpy as np


class FundusPreprocessor:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def crop_background(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        return image[y:y+h, x:x+w]

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def resize(self, image):
        return cv2.resize(image, (self.image_size, self.image_size))

    def process(self, image):
        image = self.crop_background(image)
        image = self.enhance_contrast(image)
        image = self.resize(image)
        return image
