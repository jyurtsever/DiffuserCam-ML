# DiffuserCam-ML
Machine Learning techniques applied to enhance DiffuserCam image quality and classify DiffuserCam images using PyTorch 




# For training diffusercam image classifier

```
usage: train_imagenet.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               [-use_forward_trans] [-psf_file] [-use_le_admm] [-use_random_loc_trans] [-train_admm]
               DIR


positional arguments:
  DIR                   path to dataset

optional arguments:
  
  -h, --help            show this help message and exit
  
  -use_forward_trans    runs simulated DiffuserCam forward model
  
  -psf_file             path to psf file for diffusercam forward model
  
  -use_le_admm          runs learned admm pretrained for 10 iterations in ensemble with classifier
  
  -train_admm           allows the admm model parameters to be learned during training
  
  -use_random_loc_trans augments dataset by shrinking image and moving it around frame (before running forward model)
  
  --arch ARCH, -a ARCH  classifier model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
                        
  -j N, --workers N     number of data loading workers (default: 4)
  
  --epochs N            number of total epochs to run
  
  --start-epoch N       manual epoch number (useful on restarts)
  
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
                        
  --momentum M          momentum
  
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
                        
  --print-freq N, -p N  print frequency (default: 10)
  
  --resume PATH         path to latest checkpoint (default: none)
  
  -e, --evaluate        evaluate model on validation set
  
  --pretrained          use pre-trained model
  
  --world-size WORLD_SIZE
                        number of nodes for distributed training
                        
  --rank RANK           node rank for distributed training
  
  --dist-url DIST_URL   url used to set up distributed training
  
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  
  --gpu GPU             GPU id to use.
  
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
```





### Examples:

Train classifier on imagenet dataset of normal images: 
```
cd classif
python3 train_imagenet.py image_net_directory -save_path path_to_save_your_model
```

Train classifier on imagenet dataset of DiffuserCam images: 
```
python3 train_imagenet.py image_net_directory -save_path path_to_save_your_model -use_forward_trans
```

Train classifier on imagenet dataset of DiffuserCam images with data augmentation: 
```
python3 train_imagenet.py image_net_directory -save_path path_to_save_your_model -psf_file path_to_psf -use_forward_trans -use_random_loc_trans
```

Train classifier on imagenet dataset of DiffuserCam images with 10 iterations of learned reconstruction: 
```
python3 train_imagenet.py image_net_directory -save_path path_to_save_your_model -psf_file path_to_psf -use_forward_trans
```

Train classifier on imagenet dataset of DiffuserCam images with ten iterations of pretrained admm: 
```
python3 train_imagenet.py image_net_directory -save_path path_to_save_your_model -psf_file path_to_psf -use_le_admm
```

Train classifier on imagenet dataset of DiffuserCam images with ten iterations: 
```
python3 train_imagenet.py image_net_directory -save_path path_to_save_your_model -psf_file path_to_psf -use_le_admm
```


# For Running Live Classification 
1. Using Webcam video from laptop
  - In server (can be over ssh): 
  ```
  usage: server.py [-model_dir] [-imagenet_train_dir] [--pretrained] [--port]
  
  Arguments:
    
    -model_dir (optional)              directory with pretrained model .pth or pt file 
    -imagenet_train_dir (optional)     directory containing imagenet training data - used to tell the order of classes
                                       if argument not provided, will use default 1000 imagenet classes
    
    --pretrained (optional)            will use pretrained model from internet with 1000 imagenet classes
    --port (optional)                  specifies the port you would like to use (default 8090)
    
  ```
  Examples:
  
  Classify your webcam images using a model you trained on some imagenet images:
  ```
  python3 server.py -model_dir path/to/saved/model.pth -imagenet_train_dir path/to/imagenet/training/images/folder
  ```
  
  Classify webcam images using a model trained on the internet on 1000 imagenet classes:
  
  ```
  python3 server.py --pretrained
  ```
    
  
  - In webcam_client (on your laptop)
  ```
    usage: webcam_client.py IP_ADDRESS [--port]
  
  Positional Arguments:
    IP_ADDRESS           the ip address of the server you are connecting to
  
  Optional Arguments:
    --port               the port you would like to connect to (default 8090)
  ```
  
  Example:
  
  Run the webcam on your laptop connecting to server with ip address 555.55.555.55
  ```
  python3 webcam_client.py 555.55.555.55
  ```

2. Using DiffuserCam
  - In server (over ssh): 
    
    ```
    python3 server_with_recon.py path_to_model.pth.tar  -imagenet_train_dir path_to_directory_with_imagenet_training_data -psf_file path_to_psf_file -recon_iters 10
    ```
  - In client (laptop connected to DiffuserCam)
    
    ```
    python3 pylon_client_with_recon.py
    ```
