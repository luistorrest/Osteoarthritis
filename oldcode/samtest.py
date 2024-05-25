from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
model_path=r"C:\Users\grupo_gepar\Downloads\sam_vit_h_4b8939.pth"
model_type="vit_h"
path_image=r"C:\Users\grupo_gepar\Documents\lucho\Osteo\archivePrep\test\0\9392060L.png"
#path_image=r"C:\Users\grupo_gepar\Documents\lucho\Osteo\DatasetPrep\train\train\Osteoarthritis\9645958L.png"
sam = sam_model_registry[model_type](checkpoint=model_path)
mask_generator = SamAutomaticMaskGenerator(sam)


import matplotlib.pyplot as plt
import cv2

import numpy as np

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

image = cv2.imread(path_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()
plt.close("all")

masks = mask_generator.generate(image)
for i, (mask) in enumerate(zip(masks)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask[0]["segmentation"], plt.gca())
    plt.title(f"Mask {i+1}", fontsize=18)
    plt.axis('off')
    plt.show()  

hueso_masks = []
for i, (mask) in enumerate(zip(masks)):
    mask_array=mask[0]["segmentation"]
    
    plt.figure(figsize=(10,10))
    plt.imshow(mask_array)
    plt.axis('off')

    mask_array.shape
    middle_x=mask_array.shape[0]//2
    middle_y=mask_array.shape[1]//2

    hueso_up=np.sum(mask_array[:middle_y,middle_x])/mask_array[:middle_y,middle_x].shape
    hueso_down=np.sum(mask_array[middle_y:,middle_x])/mask_array[middle_y:,middle_x].shape

    hueso= hueso_up > 0.3 or hueso_down > 0.3
    plt.title(f"Mask {i+1} hueso {hueso} {hueso_up} {hueso_down}", fontsize=18)
    plt.show()
    if hueso:
        hueso_masks.append(mask_array)


if len(hueso_masks) != 2:
    pass
    # TODO: Further logic to distinguish huesos

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(gray_image)

for i,hmask_ in enumerate(hueso_masks):
    equalized_image[hmask_] = 255

full_mask = np.logical_or(hueso_masks[0],hueso_masks[1])

equalized_image[full_mask] = 255
equalized_image[np.logical_not(full_mask)]=0

plt.figure(figsize=(10,10))
plt.imshow(equalized_image,cmap="gray")
plt.axis('on')
plt.show()
plt.close("all")