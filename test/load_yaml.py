
from easydict import EasyDict as edict


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
        print(yaml_cfg)



if __name__ == '__main__':
    cfg_from_file("../experiments/cfgs/faster_rcnn_end2end.yml")
