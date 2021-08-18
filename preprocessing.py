import cv2
import time

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import random
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import random


class ShapeAverager:

    def _get_polygon(self):
        used_vertices = 4#max(3, int(random.gauss(self.vertices, 2.0)))
        image = Image.new("P", (self.shape_size, self.shape_size))
        draw = ImageDraw.Draw(image)

        # generate images
        points = np.random.randint(0, self.shape_size-1, 2 * used_vertices).reshape(-1, 2)
        points = [tuple(p) for p in tuple(points)]
        draw.polygon(points, fill=1)
        mask = np.array(image)
        return mask

    def __init__(self, shape_size=50, vertices=4, average_number_of_shapes=4):
        self.shape_size = shape_size
        self.vertices = vertices
        self.average_number_of_shapes = average_number_of_shapes

    def __call__(self, img):
        number_of_shapes = max(0, int(random.gauss(self.average_number_of_shapes, 2.0)))
        masks = [self._get_polygon() for i in range(number_of_shapes)]

        # get location of shapes
        xs = np.random.randint(0, img.shape[0]-1 - self.shape_size, number_of_shapes)
        ys = np.random.randint(0, img.shape[1]-1 - self.shape_size, number_of_shapes)
        points = np.dstack((xs, ys))[0]

        for p, mask in zip(points, masks):
            # calculate average mask
            values = img[p[0]:p[0]+self.shape_size, p[1]:p[1]+self.shape_size]
            changed_values = values[mask > 0].T
            average_color = changed_values.sum(axis=1) / float(mask.sum())
            values[mask > 0] = average_color.astype(int)

        return img


class QuantizationBlob:

    MS_PAINT_COLORS = [
        (90, 90, 90),  # gray
        (136, 0, 21),
        (237, 28, 36),
        (255, 127, 39),
        (255, 242, 0),
        (34, 177, 76),
        (0, 162, 232),
        (63, 72, 204),
        (163, 73, 164),
        (200, 200, 200),
        (195, 195, 195),
        (185, 122, 87),
        (255, 174, 201),
        (255, 201, 14),
        (239, 228, 176),
        (181, 230, 29),
        (153, 217, 234),
        (112, 146, 190),
        (200, 191, 131),
        (50, 50, 50),
        (30, 30, 70),
        (30, 70, 30),
        (70, 30, 30),
        (129, 167, 68),
        (226, 178, 158),
    ]

    def _quantize_image(self, img, colors):
        n_colors = len(colors)
        arr = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_colors).fit(arr[:50])
        colors = np.array(colors)
        kmeans.cluster_centers_ = colors.astype(float)
        if arr.dtype == float or arr.dtype == 'float32':
            arr = (arr * 255).astype('uint8')
        labels = kmeans.predict(arr)
        less_colors = colors[labels].reshape(img.shape)
        return less_colors

    def __init__(self, blurs=2, blur_kernel_size=5, color_removal_percentage=0.03):
        self.blurs = blurs
        self.blur_kernel_size = blur_kernel_size
        self.color_removal_percentage = color_removal_percentage

    def __call__(self, sample):
        colors = self.MS_PAINT_COLORS
        used_colors = [color for color in colors if random.random() > self.color_removal_percentage]
        for i in range(self.blurs):
            sample = cv2.blur(sample, ksize=(self.blur_kernel_size, self.blur_kernel_size))

        return self._quantize_image(sample, used_colors)