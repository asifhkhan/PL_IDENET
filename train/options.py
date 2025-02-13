
import sys

import yaml

try:
    sys.path.append("../../")
    from utils_blur import OrderedYaml
except ImportError:
    pass

Loader, Dumper = OrderedYaml()


def parse(opt_path):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)


    return opt



