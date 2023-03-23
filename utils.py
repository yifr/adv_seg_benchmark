import os
import cv2
from glob import glob
import numpy as np
from PIL import Image
from colormap import recolor
import matplotlib.pyplot as plt 


def load_model_predictions(dataset, image_id, model_name, recolor_mask=True, size=(512, 512)):
    pred_path = os.path.join("model_predictions", model_name, dataset) + "/*"
    images = glob(pred_path)
    images.sort()
    if model_name == "EISEN":
        mask = np.load(open(images[image_id], "rb")).astype(np.uint8)
        mask = np.array(Image.fromarray(mask).resize(size, resample=Image.Resampling.NEAREST))
    else:
        mask = cv2.imread(images[image_id])
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    
    if recolor_mask:
        mask = recolor(mask)

    return mask

# Load an image and ground truth masks
def load_input(dataset, image_id, recolor_mask=True, size=(512, 512)):
    image_path = os.path.join("data", dataset, "images", f"image_{image_id:03d}.png")
    mask_path = os.path.join("data", dataset, "masks", f"mask_{image_id:03d}.png")
    task_image = np.array(Image.open(image_path).resize(size, resample=Image.Resampling.BILINEAR))
    mask = np.array(Image.open(mask_path).convert("L").resize(size, resample=Image.Resampling.NEAREST))

    if recolor_mask:
        mask = recolor(mask)

    return task_image, mask

def draw_circle(ax, point, color, labels=None, radius=8):
    circle = plt.Circle(point, radius=radius, color=color, label=f"Label: {labels[point[1], point[0]]}")
    ax.add_patch(circle)
    return 

def plot_pair(img_a, img_b, labels=[], probe_locs=[]):
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    axs[0].imshow(img_a)
    axs[1].imshow(img_b)
    if len(labels) == 2:
        axs[0].set_title(labels[0])
        axs[1].set_title(labels[1])
    elif len(labels) == 1:
        plt.suptitle(labels[0]) 

        
    if probe_locs:
        for i, ax in enumerate(axs):
            for p in probe_locs:
                p1 = p[0]
                p2 = p[1]
                if i == 0:
                    mask = img_a
                else:
                    mask = img_b

                draw_circle(ax, p1, "red", labels=mask)
                draw_circle(ax, p2, "green", labels=mask)
            
            ax.legend()


    for ax in axs:
        ax.set_xticks([], [])
        ax.set_yticks([], [])


    return fig