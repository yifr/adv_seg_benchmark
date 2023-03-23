import numpy as np

COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3) 


def recolor(mask, use_labels=None):
    """
    Colors a mask with pre-defined colormap (with nice colors)

    Args:
        mask (`np.ndarray`):
            (height x width) segmentation mask of labels (ints)
        use_labels (`list`):
            explicit labels to use for coloring 
    """
    if mask.shape[-1] == 3:
        mask, mask_labels = as_labels(mask, return_labels=True)
    else:
        mask_labels = np.unique(mask).astype(np.uint8)

    new_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    
    for index in mask_labels:
        y, x = np.where(mask == index)
        if index >= len(COLORS):
            new_mask[y, x] = np.random.random(3)
        elif use_labels is not None:
            if use_labels[index] == -1:
                new_mask[y, x] = COLORS[len(COLORS) - index - 1]
            else:
                new_mask[y, x] = COLORS[int(use_labels[index])]
        else:
            new_mask[y, x] = COLORS[int(index)]

    return new_mask

def as_labels(mask, return_labels=False):
    assert mask.shape[-1] == 3

    height, width = mask.shape[0], mask.shape[1]
    mask = mask.reshape(-1, 3)
    unique = np.unique(mask, axis=0)
    new_mask = np.zeros(height * width)

    for i, label in enumerate(unique):
        matches = np.where(np.all(np.equal(mask, label), axis=1))
        new_mask[matches] = i
    
    new_mask = new_mask.reshape((height, width))
    if return_labels:
        return new_mask, list(range(len(unique)))
    else:
        return new_mask

def relabel(mask, labels):
    assert len(mask.shape) == 2
    old_labels = np.unique(mask)
    for i in range(len(old_labels)):
        new_label = labels[i]
        if new_label < 0:
            new_label = len(labels) - i

        mask[np.where(mask == old_labels[i])] = new_label

    return mask