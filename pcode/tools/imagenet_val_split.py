import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet_dir', required=True, type=str)
parser.add_argument('--imagenet_val_index', required=True, type=str)
args = parser.parse_args()

with open(args.imagenet_val_index) as f:
    for ln in f:
        ln = ln.rstrip()
        target_name = ln.split(" ")[0]

        unclass_path = target_name.split("/")
        unclass_path = "/".join([args.imagenet_dir, unclass_path[0], unclass_path[2]])
        class_name = target_name.split("/")[1]

        target_dir = os.path.join( args.imagenet_dir, "val", class_name)
        path = os.path.join( args.imagenet_dir, target_name)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        os.rename(unclass_path, path)
