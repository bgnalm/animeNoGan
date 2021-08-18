import torch
import models
import preprocessing
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import dataset
import train
import csv
import os


def save_data(filepath, train_loss, val_loss):
    with open(filepath, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        for i in range(len(train_loss)):
            row = {
                'epoch': i+1,
                'train_loss': train_loss[i],
                'val_loss': val_loss[i]
            }
            writer.writerow(row)


OUTPUT_DIR = os.curdir


def run_test(test_name, model, criterion, optimizer, lr_scheduler, train_dataloader, val_dataloader, epochs, notebook=True, outdir=OUTPUT_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = train.Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        notebook=notebook)
    train_loss, val_loss, _ = trainer.run_trainer()
    output_dir = os.path.join(outdir, test_name)
    os.makedirs(output_dir, exist_ok=True)
    save_data(os.path.join(output_dir, "results.csv"), train_loss, val_loss)


if __name__ == '__main__':
    model = models.get_model()
    video_dataset = dataset.VideoDataset(
        '../dragon ball/EP.70.480p.mp4',
        initial_transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]),
        information_loss_transform=transforms.Compose([])
    )

    run_test(
        test_name="vanilla",
        model=model,
        criterion=torch.nn.L1Loss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        train_dataloader= DataLoader(video_dataset, batch_size=4, shuffle=False),
        val_dataloader=None,
        lr_scheduler=None,
        epochs=3,
        notebook=False
    )




