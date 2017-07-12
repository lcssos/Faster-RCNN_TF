# coding=utf-8
import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse

def _load_pascal_annotation(index):
    _classes = ('__background__',  # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
    _class_to_ind = dict(zip(_classes, range(len(_classes))))

    filename = os.path.join('../data/VOCdevkit2007/VOC2007', 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 21), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        # 获取物体分类号
        cls = _class_to_ind[obj.find('name').text.lower().strip()]
        # 一个图片中标记的多个物体的数组
        # 第ix个物体对应的两点坐标
        boxes[ix, :] = [x1, y1, x2, y2]
        # 物体的分类号
        gt_classes[ix] = cls
        # 稀疏矩阵，第ix个物体的第cls分类位置标记为1，其余为0
        overlaps[ix, cls] = 1.0
        # 物体区域面积
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)
    print(overlaps)
    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

if __name__ == '__main__':
    r = _load_pascal_annotation('000001')
    print(r)
