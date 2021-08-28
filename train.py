import numpy as np
import torch
import dataset
import os
from PIL import Image

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_dataloader,
                 val_dataloader,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.val_dataloader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.val_dataloader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.train_dataloader), 'Training', total=len(self.train_dataloader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.val_dataloader), 'Validation', total=len(self.val_dataloader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()


    def _unnormalize_output_image(self, img):
        # assuming x and y are Batch x 3 x H x W and mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
        mean = [0.49031248, 0.5466465, 0.48358064]
        std = [0.25759599, 0.23144381, 0.25116349]
        x = img.new(*img.size())
        x[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
        x[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
        x[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
        return x

    def calculate_example_images(self, outdir, global_transform, unnormalize=False):
        d = dataset.ExampleImagesDataset(global_transform)
        self.model.eval()  # evaluation mode

        for i, (x, y) in enumerate(d):
            input, target = x.to(self.device), y  # send to device (GPU or CPU)
            with torch.no_grad():
                new_input = input.reshape(1, *input.shape)
                out = self.model(new_input)
                out_cpu = out.cpu()
                out_normalized = self._unnormalize_output_image(out_cpu)
                out_normalized = out_normalized.permute(0, 2, 3, 1)
                array = np.array(out_normalized)
                out_image = array[0]
                out_image = (out_image * 255.0).astype('uint8')
                img = Image.fromarray(out_image)
                img.save(os.path.join(outdir, target))




