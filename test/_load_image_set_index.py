
import os

def _load_image_set_index():
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join('../data/VOCdevkit2007/VOC2007', 'ImageSets', 'Main', 'trainval.txt')
    assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index


if __name__ == '__main__':
    image_index = _load_image_set_index()
    print(image_index)