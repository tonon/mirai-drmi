import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, t1=100, t2=200):
        self.t1 = t1
        self.t2 = t2
        
    def detect(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, self.t1, self.t2)
