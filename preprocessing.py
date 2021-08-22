import os

import cv2
import time

import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
import numpy as np
import random
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import random
import torchvision.transforms
from tqdm import tqdm


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
        self.to_tensor = torchvision.transforms.PILToTensor()

    def __call__(self, sample):
        colors = self.MS_PAINT_COLORS
        used_colors = [color for color in colors if random.random() > self.color_removal_percentage]
        img = sample
        if type(sample) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        for i in range(self.blurs):
            img = cv2.blur(img, ksize=(self.blur_kernel_size, self.blur_kernel_size))

        final_img = self._quantize_image(img, used_colors)
        return torch.Tensor(final_img).permute(2, 0, 1)

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
@ignore_warnings(category=ConvergenceWarning)
def preprocess_video(vid_input, vid_output, new_size):
    prec = QuantizationBlob(blurs=2, blur_kernel_size=5, color_removal_percentage=0)
    capture = cv2.VideoCapture(vid_input)
    output = cv2.VideoWriter(vid_output, -1, 1, (new_size, new_size))
    success, frame = capture.read()

    for i in tqdm(range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
        new_image = Image.fromarray(frame).resize((new_size, new_size))
        quant = prec(np.array(new_image))
        output.write(quant.numpy())

    capture.release()
    output.release()

@ignore_warnings(category=ConvergenceWarning)
def preprocess_video_to_images(vid_input, vid_output_dir, new_size, skip_factor=15, first_index=0, intro_outro_percentage=0.075):
    prec = QuantizationBlob(blurs=2, blur_kernel_size=5, color_removal_percentage=0)
    capture = cv2.VideoCapture(vid_input)
    success, frame = capture.read()

    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    intro_outro_frames = int(number_of_frames * intro_outro_percentage)
    for i in range(intro_outro_frames):
        success, frame = capture.read()

    count = first_index
    for i in tqdm(range(number_of_frames - (2 * intro_outro_frames))):
        if i % skip_factor == 0:
            new_image = np.array(Image.fromarray(frame).resize((new_size, new_size)))
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            quant = prec(np.array(new_image))
            new_image = Image.fromarray(quant.permute(1, 2, 0).numpy().astype('uint8'))
            new_image.save(os.path.join(vid_output_dir, f'{count}.jpg'))
            count += 1

        success, frame = capture.read()

    capture.release()
    return count

@ignore_warnings(category=ConvergenceWarning)
def preprocess_video_to_gt_images(vid_input, vid_output_dir, new_size, skip_factor=15, first_index=0, intro_outro_percentage=0.075):
    capture = cv2.VideoCapture(vid_input)
    success, frame = capture.read()

    number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    intro_outro_frames = int(number_of_frames * intro_outro_percentage)
    for i in range(intro_outro_frames):
        success, frame = capture.read()

    count = first_index
    for i in tqdm(range(number_of_frames - (2 * intro_outro_frames))):
        if i % skip_factor == 0:
            new_image = np.array(Image.fromarray(frame).resize((new_size, new_size)))
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            new_image = Image.fromarray(new_image)
            new_image.save(os.path.join(vid_output_dir, f'{count}.jpg'))
            count += 1

        success, frame = capture.read()

    capture.release()
    return count

import random
def fix_train_val():
    files = os.listdir('./dragon_ball_gt//train')
    to_move = set()
    while len(to_move) < 1500:
        to_move.add(random.choice(files))

    for f in tqdm(to_move):
        os.rename(os.path.join('./dragon_ball_gt', 'train', f), os.path.join('./dragon_ball_gt', 'val', f))
        os.rename(os.path.join('./dragon_ball_preprocessed', 'train', f), os.path.join('./dragon_ball_preprocessed', 'val', f))

    def fix_file_indices(dir_path):
        files_in_dir = os.listdir(dir_path)
        indices = [int(f[:-4]) for f in files_in_dir]
        indices.sort()
        counter = 0
        for idx in tqdm(indices):
            os.rename(os.path.join(dir_path, f'{idx}.jpg'), os.path.join(dir_path, f'{counter}.jpg'))
            counter += 1

    fix_file_indices(os.path.join('./dragon_ball_gt', 'train'))
    fix_file_indices(os.path.join('./dragon_ball_gt', 'val'))
    fix_file_indices(os.path.join('./dragon_ball_preprocessed', 'train'))
    fix_file_indices(os.path.join('./dragon_ball_preprocessed', 'val'))




if __name__ == '__main__':
    source = "./dragon ball"
    dest = './dragon_ball_preprocessed'
    files = os.listdir(source)
    first_index = 0
    for f in files:
        first_index = preprocess_video_to_images(os.path.join(source, f), dest, 224, first_index=first_index)



