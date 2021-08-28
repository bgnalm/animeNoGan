import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms

def main():
    count = 0
    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    t = transforms.ToTensor()
    for (dir_path, dir_names, filenames) in os.walk('./dragon_ball_gt'):
        for f in filenames:
            if count % 1000 == 0:
                print(count)
            full_filename = os.path.join(dir_path, f)
            img = Image.open(full_filename)
            img_tensor = t(img)
            mean_sum += np.array([img_tensor[0].mean(), img_tensor[1].mean(), img_tensor[2].mean()])
            std_sum += np.array([img_tensor[0].std(), img_tensor[1].std(), img_tensor[2].std()])
            count += 1

    print(mean_sum / count)
    print(std_sum / count)


if __name__ == '__main__':
    main()