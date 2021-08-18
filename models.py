import segmentation_models_pytorch as smp
import torch

def get_model():
     model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    )

     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     model.to(device)
     model.eval()
     return model
