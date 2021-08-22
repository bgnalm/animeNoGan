import torchvision.transforms.functional as tf
from torchvision.transforms import ColorJitter
import random

class RandomTransform:

    def __init__(self, flip=0.5, rotate=0.2, color_jitter=False):
        self.flip = flip
        self.rotate = rotate
        self.jitter_transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.color_jitter = color_jitter

    def __call__(self, img, gt_img):
        if random.random() > self.flip:
            img = tf.hflip(img)
            gt_img = tf.hflip(gt_img)

        if self.color_jitter:
            img = self.jitter_transform(img)

        if random.random() > self.rotate:
            rotation = (random.random() - 0.5) * 10
            img = tf.rotate(img, angle=rotation)
            gt_img = tf.rotate(gt_img, angle=rotation)

        return img, gt_img
