import argparse
import os
import random
import shutil


def main(args):
    train_dir = os.path.join(args.root, 'train')
    val_dir = os.path.join(args.root, 'val')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)
    walk = list(os.walk(args.root))
    for path, subdirs, files in walk:
        sub_folder = os.path.basename(path)

        print(sub_folder)
        if not sub_folder or sub_folder == 'train' or sub_folder == 'val':
            continue

        if not os.path.isdir(os.path.join(val_dir, sub_folder)):
            os.mkdir(os.path.join(val_dir,  sub_folder))

        random.shuffle(files)
        val_files = files[:int(len(files) * args.frac_validation)]
        for val_file in val_files:
            shutil.move(os.path.join(path, val_file), os.path.join(val_dir, sub_folder, val_file))

        shutil.move(path, train_dir)


if __name__ == '__main__':
    random.seed(123)
    parser = argparse.ArgumentParser(description='Makes a train and validation folder in dataset')
    parser.add_argument('-root', type=str)
    parser.add_argument('-frac_validation', type=float, default=.08)
    args = parser.parse_args()

    gVars = {}
    main(args)
