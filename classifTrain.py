from resnet import *
import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--train_names", type=str)
    parser.add_argument("--test_names", type=str)
    args = parser.parse_args()
    main(args)
