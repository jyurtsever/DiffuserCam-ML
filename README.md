# DiffuserCam-ML
Machine Learning techniques applied to enhance DiffuserCam image quality and classify DiffuserCam images using PyTorch 


Examples:

Train classifier on imagenet dataset of normal images: 

python3 train_imagenet.py [image_net_directory] -save_path [path_to_save_your_model]


Train classifier on imagenet dataset of DiffuserCam images: 

python3 train_imagenet.py [image_net_directory] -save_path [path_to_save_your_model] -use_forward_trans


Train classifier on imagenet dataset of DiffuserCam images with data augmentation: 

python3 train_imagenet.py [image_net_directory] -save_path [path_to_save_your_model] -use_forward_trans -use_random_loc_trans


Train classifier on imagenet dataset of DiffuserCam images with 10 iterations of learned reconstruction: 

python3 train_imagenet.py [image_net_directory] -save_path [path_to_save_your_model] -use_forward_trans


Train classifier on imagenet dataset of DiffuserCam images with ten iterations of learned admm: 

python3 train_imagenet.py [image_net_directory] -save_path [path_to_save_your_model] -use_le_admm



