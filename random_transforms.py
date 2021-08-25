import torchvision.transforms.functional as tf
from torchvision.transforms import ColorJitter, GaussianBlur
import random

class RandomTransform:

    def __init__(self, flip=0.5, rotate=0.2, color_jitter=False, blur_operations_avg=0.5, return_gt_image_percent=0.05):
        self.flip = flip
        self.rotate = rotate
        self.jitter_transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.blur = GaussianBlur(kernel_size=5)
        self.color_jitter = color_jitter
        self.return_gt_image_percent = return_gt_image_percent
        self.blur_operations_avg = blur_operations_avg


    def __call__(self, img, gt_img):
        if random.random() > self.flip:
            img = tf.hflip(img)
            gt_img = tf.hflip(gt_img)

        if self.color_jitter:
            img = self.jitter_transform(img)

        blur_operations = int(max(random.gauss(mu=self.blur_operations_avg, sigma=0.75), 0))
        for i in range(blur_operations):
            img = self.blur(img)

        if random.random() > self.rotate:
            rotation = (random.random() - 0.5) * 10
            img = tf.rotate(img, angle=rotation)
            gt_img = tf.rotate(gt_img, angle=rotation)

        if random.random() < self.return_gt_image_percent:
            return gt_img, gt_img

        return img, gt_img
