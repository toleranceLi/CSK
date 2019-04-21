import numpy as np
class RGB2GRAY:
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])