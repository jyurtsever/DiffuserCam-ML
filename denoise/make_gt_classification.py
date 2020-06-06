import argparse
import json

filenames = ['baby_r1.txt',  'bird_r1.txt', 'car_r1.txt', 'clouds_r1.txt', 'dog_r1.txt',
             'female_r1.txt',  'flower_r1.txt',  'male_r1.txt',  'night_r1.txt', 'people_r1.txt',
             'portrait_r1.txt',  'river_r1.txt',  'sea_r1.txt',  'tree_r1.txt']
num_images = 25000

def main(args):
    num_class = len(filenames)
    gt = [[0 for _ in range(num_class)] for _ in range(0, num_images + 1)]
    for i, name in enumerate(filenames):
        f = open(args.ann_dir + name, 'r')
        for line in f:
            gt[int(line.strip())][i] = 1
        f.close()

    data = {}
    data['gt'] = []
    for im_num, class_lst in enumerate(gt):
        im_filename = "im{:05}.tiff".format(im_num)
        data['gt'].append({im_filename : class_lst})
    with open(args.save_dir + 'gt_classif.txt', 'w') as outfile:
        json.dump(data, outfile)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument(
        "--ann_dir",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
    )
    parser.add_argument(
        "--save_dir",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
    )
    args = parser.parse_args()
    main(args)
