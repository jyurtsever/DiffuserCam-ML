import argparse
import os
import random
import shutil


def main(args):
    train_dir = args.root + 'train'
    val_dir = args.root + 'val'
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)

    for path, subdirs, files in os.walk(args.root):
        sub_folder = os.path.basename(path)

        print(sub_folder)
        if not sub_folder:
            continue

        if not os.path.isdir(val_dir + sub_folder):
            os.mkdir(val_dir + sub_folder)

        random.shuffle(files)
        val_files = files[:int(len(files) * args.frac_validation)]
        for val_file in val_files:
            shutil.move(path + '/' + val_file, val_dir + '/' + val_file)

        shutil.move(path, train_dir)


if __name__ == '__main__':
    random.seed(123)
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('root', type=str)
    parser.add_argument('-frac_validation', type=float, default=.08)
    args = parser.parse_args()

    gVars = {}
    main(args)
