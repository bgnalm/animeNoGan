from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import os
from PIL import Image


class VideoDataset(Dataset):

    def _restart_video(self):
        self.capture.release()
        self.capture = cv2.VideoCapture(self.video_file)
        self.last_request_idx = -1

    def __init__(self, video_file, initial_transform, information_loss_transform, skip_factor=1):
        self.video_file = video_file
        self.capture = cv2.VideoCapture(video_file)
        self.initial_transform = initial_transform
        self.information_loss_transform = information_loss_transform
        self.last_request_idx = -1
        self.skip_factor = skip_factor

    def __len__(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) // self.skip_factor

    def __getitem__(self, idx):
        if idx == 0:
            self._restart_video()

        assert idx == (self.last_request_idx + 1)
        self.last_request_idx = idx
        for i in range(self.skip_factor):
            success, frame = self.capture.read()
        assert success
        frame = Image.fromarray(frame)
        input_frame = self.initial_transform(frame)
        output_frame = self.information_loss_transform(input_frame)
        return input_frame, output_frame


def build_video_datasets(videos, initial_transforms, info_loss_transforms, skip_factor=None):
    all_datasets = [VideoDataset(vid, initial_transforms, info_loss_transforms, skip_factor) for vid in videos]
    return ConcatDataset(all_datasets)


class ImageCoupleDataset(Dataset):

    def __init__(self, image_dir, gt_dir, transforms, random_transforms=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.number_of_images = len(os.listdir(image_dir))
        self.transforms = transforms
        self.random_transforms = random_transforms

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, idx):
        train_image = Image.open(os.path.join(self.image_dir, f'{idx}.jpg'))
        gt_image = Image.open(os.path.join(self.gt_dir, f'{idx}.jpg'))
        train_image = self.transforms(train_image)
        gt_image = self.transforms(gt_image)
        if self.random_transforms is not None:
            train_image, gt_image = self.random_transforms(train_image, gt_image)

        return train_image, gt_image


def build_image_couple_dataset(transforms, random_transforms, x_dir='dragon_ball_preprocessed', y_dir='dragon_ball_gt'):
    train = ImageCoupleDataset(
        os.path.join(x_dir, 'train'),
        os.path.join(y_dir, 'train'),
        transforms=transforms,
        random_transforms=random_transforms
    )

    val = ImageCoupleDataset(
        os.path.join(x_dir, 'val'),
        os.path.join(y_dir, 'val'),
        transforms=transforms
    )

    return train, val


class ExampleImagesDataset(Dataset):

    def __init__(self, initial_transform):
        self.initial_transform = initial_transform
        self.files = os.listdir('./example_images/')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        frame = Image.open(os.path.join('./example_images', self.files[idx]))
        frame = frame.convert('RGB')
        return self.initial_transform(frame), self.files[idx]


