import numpy as np
from patchify import patchify
import cv2
import glob

image_names = glob.glob('C:\\Users\\rahil\\PycharmProjects\\thesis\\creating_dataset\\data\\images\\*.png')
image_names.sort()

mask_names = glob.glob('C:\\Users\\rahil\\PycharmProjects\\thesis\\creating_dataset\\data\\masks\\*.png')
mask_names.sort()

images = np.array([cv2.imread(file, 0) for file in image_names])
masks = np.array([cv2.imread(file, 0) for file in mask_names])

for img in range(images.shape[0]):
    large_image = images[img]
    patches_img = patchify(large_image, (256, 256), step=256)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]

            cv2.imwrite('C:\\Users\\rahil\\PycharmProjects\\thesis\\creating_dataset\\dataset\\train_images\\train\\'
                        + 'image_'
                        + str(img)
                        + '_'
                        + str(i) + str(j)
                        + ".png",
                        single_patch_img)

for msk in range(masks.shape[0]):
    large_mask = masks[msk]
    patches_mask = patchify(large_mask, (256, 256), step=256)

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]

            cv2.imwrite('C:\\Users\\rahil\\PycharmProjects\\thesis\\creating_dataset\\dataset\\train_masks\\train\\'
                        + 'mask_'
                        + str(msk)
                        + '_'
                        + str(i) + str(j)
                        + ".png",
                        single_patch_mask)
