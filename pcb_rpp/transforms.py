import cv2
import numpy as np
from PIL import Image


class Resize:
    def __init__(self, target_size=(384, 128)):
        """
        :param target_size: a tuple (target_h, target_w)
        """
        self.target_size = target_size

    def __call__(self, img):
        """
        :param img: a numpy.ndarray matrix
        :return: resized image
        """

        h, w, c = img.shape
        ratio = min(self.target_size[0] / h, self.target_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        new_h, new_w = max(new_h, 1), max(new_w, 1)
        try:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(new_h, new_w, h, w)
            raise e

        pad_left = int((self.target_size[1] - new_w) / 2)
        pad_right = self.target_size[1] - new_w - pad_left
        pad_bottom = self.target_size[0] - new_h
        img = np.pad(img, ((0, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
        assert img.shape[:2] == self.target_size
        return img


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = np.flip(img, axis=1).copy()
        return img


class Compose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img
