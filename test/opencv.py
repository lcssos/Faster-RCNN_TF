
import _init_paths
from fast_rcnn.config import cfg
import os, cv2
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

image_name = "004545.jpg"

img_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)

im = cv2.imread(img_file)

print(type(im))

# im = im[:, :, (2, 1, 0)]
# fig, ax = plt.subplots(figsize=(12, 12))
# ax.imshow(im, aspect='equal')
#
# plt.axis('off')
# plt.tight_layout()
# plt.draw()

plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# cv2.imshow('image',im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()