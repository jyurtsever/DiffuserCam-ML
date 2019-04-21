
#from admm_rgb import *
#from model_unrolled_layered import *

# import model_color as my_model_color
import progressbar
import scipy
import sys
from ADMM_tf_utils import *

def main():
    bar = progressbar.ProgressBar(maxval=num_photos, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    csv_contents = pd.read_csv(csv_file_path)

    for i in range(start, min(start + num_photos, len(csv_contents))):
        im_name = csv_contents.iloc[i,0]
        im, gt_im = read_and_downsample_im(im_name, im_name, down_sizing)
        scipy.misc.imsave(save_path + im_name, im)
        bar.update(i)
    bar.finish()




if __name__ == '__main__':
    csv_file_path = '../mirflickr25k/filenames.csv'
    num_photos = int(sys.argv[1])
    num_iters = int(sys.argv[2])
    start = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    #####   SAVE FILE    #####
    save_path = sys.argv[5]
    down_sizing = 4
    main()
