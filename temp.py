import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

input_folder = '/content/U-2-Net/dataset/Image'
gt_folder = '/content/U-2-Net/car_virtual_bg/ground_truth'
new_trained = '/content/U-2-Net/car_virtual_bg/u2net_dice_200'

for filename in os.listdir(input_folder):
    name, ext = os.path.splitext(filename)

    input_path = os.path.join(input_folder, filename)
    gt_path = os.path.join(gt_folder, name + '.png')
    mask_path = os.path.join(new_trained, name + '.png')

    input_img = plt.imread(input_path)
    gt_img = plt.imread(gt_path)
    mask_img = cv2.imread(mask_path)
    orientation = detect_car_orientation(mask_img)


    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    axs[0].imshow(input_img)
    axs[0].set_title('Input')
    axs[1].imshow(gt_img, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(mask_img)
    axs[2].set_title('Prediction')
    # axs[3].axis('off')
    axs[3].text(0.5, 0.5, orientation,
          horizontalalignment='center',
          verticalalignment='center',
          fontsize=14)
    axs[3].set_title('Orientation')
    plt.tight_layout()

    plt.show()