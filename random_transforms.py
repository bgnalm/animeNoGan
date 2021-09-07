import numpy as np
import torch
import torchvision.transforms.functional as tf
from torchvision.transforms import ColorJitter, GaussianBlur, RandomErasing, RandomAdjustSharpness
from skimage.transform import PiecewiseAffineTransform, warp
import random


class RandomTransform:

    def _grid_sample(self, tensor_image):
        # get random movement
        image_size = tensor_image.shape[1]
        interpolation_points = 10
        changed_vectors = torch.normal(mean=0, std=0.1, size=(interpolation_points-1, interpolation_points-1, 2))

        # Create linear grid
        d = torch.linspace(-1, 1, interpolation_points)
        meshx, meshy = torch.meshgrid((d, d))
        grid = torch.stack((meshy, meshx), 2)

        grid[0:-1, 0:-1, :] += changed_vectors
        grid = grid.unsqueeze(0)  # add batch dim

        output = torch.nn.functional.grid_sample(tensor_image.unsqueeze(0), grid, align_corners=False)
        return output[0]

    def _piecewise_affine(self, tensor_image, interpolation_points=20, std=5.0):
        # covert to numpy
        image = tensor_image.permute(1, 2, 0).numpy()
        rows, cols = image.shape[0], image.shape[1]

        src_cols = np.linspace(0, cols, interpolation_points)
        src_rows = np.linspace(0, rows, interpolation_points)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]

        dst = src + np.random.normal(0, std, src.shape)
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        out = warp(image, tform, output_shape=(rows, cols))

        # convert to tensor
        return torch.tensor(out).permute(2, 0, 1)



    def __init__(self, flip=0.5, rotate=0.2, color_jitter=False, blur_operations_avg=0.0, return_gt_image_percent=0.00,
                 number_of_random_erases=0, erase_random_pixel_color=False, apply_deformation=False, deformation_points=13, deformation_std=8.0):
        self.flip = flip
        self.rotate = rotate
        self.jitter_transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.blur = GaussianBlur(kernel_size=5)
        self.color_jitter = color_jitter
        self.return_gt_image_percent = return_gt_image_percent
        self.blur_operations_avg = blur_operations_avg
        self.number_of_random_erases = number_of_random_erases
        self.random_erase = RandomErasing(p=0.5, value='random' if erase_random_pixel_color else 0)
        self.apply_deformation = apply_deformation
        self.deformation_points = deformation_points
        self.deformation_std = deformation_std


    def __call__(self, img, gt_img):
        if self.apply_deformation:
            img = self._piecewise_affine(img, self.deformation_points, self.deformation_std)
        if random.random() > self.flip:
            img = tf.hflip(img)
            gt_img = tf.hflip(gt_img)

        if self.color_jitter:
            img = self.jitter_transform(img)

        blur_operations = int(max(random.gauss(mu=self.blur_operations_avg, sigma=0.75), 0))
        for i in range(blur_operations):
            img = self.blur(img)

        for i in range(self.number_of_random_erases):
            img = self.random_erase(img)

        if random.random() > self.rotate:
            rotation = (random.random() - 0.5) * 10
            img = tf.rotate(img, angle=rotation)
            gt_img = tf.rotate(gt_img, angle=rotation)

        if random.random() < self.return_gt_image_percent:
            return gt_img, gt_img

        return img, gt_img

class RandomTransformForDeromedImages:

    def __init__(self, flip=0.5, color_jitter=False, rotate=False, return_gt_image_percent=0.0):
        self.flip = flip
        self.color_jitter = color_jitter
        self.rotate = rotate
        self.return_gt_image_percent = return_gt_image_percent

        self.jitter_transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

        self.random_blurs = [
            RandomAdjustSharpness(sharpness_factor=1, p=0.0),
            RandomAdjustSharpness(sharpness_factor=0.8, p=0.8),
            RandomAdjustSharpness(sharpness_factor=0.6, p=0.8),
            RandomAdjustSharpness(sharpness_factor=0.4, p=0.8),
        ]

        self.posterize_bits = [8, 7, 6, 5, 4]


    def __call__(self, img, gt_img):
        if random.random() < self.flip:
            img = tf.hflip(img)
            gt_img = tf.hflip(gt_img)

        blur = random.choice(self.random_blurs)
        img = blur(img)

        img = tf.posterize(img, random.choice(self.posterize_bits))

        if self.color_jitter:
            img = self.jitter_transform(img)

        if random.random() > self.rotate:
            rotation = (random.random() - 0.5) * 10
            img = tf.rotate(img, angle=rotation)
            gt_img = tf.rotate(gt_img, angle=rotation)

        if random.random() < self.return_gt_image_percent:
            return gt_img, gt_img

        return img, gt_img
