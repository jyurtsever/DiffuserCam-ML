import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
import sys
import skimage
import numpy as np
from scipy.signal import convolve2d, fftconvolve
from random import randint

sys.path.append('./models/')
import admm_model as admm_model_plain
from utils import load_psf_image, preplot




from multiprocessing import set_start_method


try:
    set_start_method('spawn')
except RuntimeError:
    pass


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-save_path', type=str, help='path to save checkpoint')

parser.add_argument("-psf_file", type=str, default= '../../recon_files/psf_white_LED_Nick.tiff')

parser.add_argument('-use_le_admm', dest='use_le_admm', action='store_true',
                    help='use learned admm and train with it')

parser.add_argument('-use_forward_trans', dest='use_forward_trans', action='store_true',
                    help='runs the forward model on dataset during training in order to use data augmentation')

parser.add_argument('-use_random_loc_trans', dest='use_random_loc_trans', action='store_true',
                    help='resize and center randomly before running forward ')

parser.add_argument('-train_admm', dest='train_admm', action='store_true',
                        help='train admm hyper parameters')
parser.add_argument('-flip_diffuser_im', dest='flip_diffuser_im', action='store_true',
                        help='flip diffusercam img while training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def make_admm_model(args):
    print("Creating Recon Model")
    my_device = 'cuda:0'

    psf_diffuser = load_psf_image(args.psf_file, downsample=1, rgb=False)

    ds = 4  # Amount of down-sampling.  Must be set to 4 to use dataset images

    print('The shape of the loaded diffuser is:' + str(psf_diffuser.shape))

    psf_diffuser = np.sum(psf_diffuser, 2)

    h = skimage.transform.resize(psf_diffuser,
                                 (psf_diffuser.shape[0] // ds, psf_diffuser.shape[1] // ds),
                                 mode='constant', anti_aliasing=True)

    var_options = {'plain_admm': [],
                   'mu_and_tau': ['mus', 'tau'],
                   }

    if args.train_admm:
        learning_options = {'learned_vars': var_options['mu_and_tau']}
    else:
        learning_options = {'learned_vars': var_options['plain_admm']}


    model = admm_model_plain.ADMM_Net(batch_size=1, h=h, iterations=10,
                                      learning_options=learning_options, cuda_device=my_device)

    le_admm = torch.load('../../saved_models/model_le_admm.pt', map_location=my_device)
    le_admm.cuda_device = my_device
    for pn, pd in le_admm.named_parameters():
        for pnn, pdd in model.named_parameters():
            if pnn == pn:
                pdd.data = pd.data

    model.tau.data = model.tau.data * 1000
    model.to(my_device)
    return model

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, admm_model, model
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.use_le_admm:
        admm_model = make_admm_model(args)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.use_le_admm:
        model = Ensemble(admm_model, model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        if args.use_le_admm:
            print("using ensemble.to()")
            model = model.to(args.gpu)
        else: 
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.use_forward_trans:
        print("using forward transformation")
        if args.use_random_loc_trans:
            print("using random locations transformation")
            trans = transforms.Compose(
                [transforms.RandomHorizontalFlip(), RandomLocAndSize(), SimForwardTrans(), transforms.ToTensor()])
        else:
            trans = transforms.Compose(
                [transforms.RandomHorizontalFlip(), CenterDisplayTrans(), SimForwardTrans(), transforms.ToTensor()])

    elif args.use_le_admm:
        if args.flip_diffuser_im:
            print("flipping diffuser images")
            trans = transforms.Compose([FlipUDTrans(), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToTensor()])

    else:
        if args.flip_diffuser_im:  
            print(" Using random shifts in diffuser image colorization ")
            trans = transforms.Compose([FlipUDTrans(),
                        transforms.ColorJitter(brightness = .5, contrast=.3, saturation=.3),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ])
        else:
            print("resizing and normalizing training data")
            trans = transforms.Compose([
                        transforms.ColorJitter(brightness = .5, contrast=.3, saturation=.3),
                        transforms.RandomRotation(45),
                        transforms.RandomResizedCrop(224), #used to be random resized crop 
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
    train_dataset = datasets.ImageFolder(
        traindir,
        trans)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.use_forward_trans:
        val_loader = torch.utils.data.DataLoader(
                            datasets.ImageFolder(valdir, trans),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    
    elif args.use_le_admm or args.flip_diffuser_im:
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([FlipUDTrans(),
                    transforms.Resize((224, 224)), transforms.ToTensor(), normalize])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=args.save_path + 'checkpoint')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # remove nan gradients
        for name, p in model.named_parameters():
            if p.grad is not None:
                p.grad[p.grad != p.grad] = 0.

        # SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Ensemble(nn.Module):
    def __init__(self, denoiser, classifier):
        super(Ensemble, self).__init__()
        self.denoiser = denoiser
        self.classifier = classifier
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            normalize])

    def forward(self, x):
        out = self.denoiser(x)
        out = self.classifier(out)
        return out

    def to(self, indevice):
        super().to(indevice)
        self.denoiser.to(indevice)
        self.denoiser.h_var.to(indevice)
        self.denoiser.h_zeros.to(indevice)
        self.denoiser.h_complex.to(indevice)
        self.denoiser.LtL.to(indevice)
        return self

class FlipUDTrans:
    def __init__(self):
        pass
    
    def __call__(self, img):
        return TF.vflip(img)


class RandomLocAndSize:
    '''Outpus image of size out_dim with image in random part of the space'''

    def __init__(self, out_dim=(270, 480)):
        self.out_dim = out_dim

    def __call__(self, img):
        res = np.zeros(self.out_dim + (3,)).astype('uint8')
        rh = res.shape[0]  # res height
        rw = res.shape[1]  # res width
        img = np.array(img)

        if img.shape[0] / img.shape[1] > self.out_dim[0] / self.out_dim[1]:  # img more vertical than psf
            img = rescale(img, height=randint(self.out_dim[0] // 1.5, self.out_dim[0] - 1))
        else:  # img more horizontal than psf
            img = rescale(img, width=randint(self.out_dim[1] // 1.5, self.out_dim[1] - 1))

        top_index = randint(0, rh - img.shape[0] - 1)
        left_index = randint(0, rw - img.shape[1] - 1)
        res[top_index: top_index + img.shape[0], left_index: left_index + img.shape[1], :] = img
        return res.astype(np.float32) / 255


class CenterDisplayTrans():
    '''Outpus image of size out_dim with image in random part of the space'''

    def __init__(self, out_dim=(270, 480)):
        self.out_dim = out_dim

    def __call__(self, img):
        res = np.zeros(self.out_dim + (3,)).astype('uint8')
        rh = res.shape[0]
        rw = res.shape[1]
        img = np.array(img)
        if img.shape[0] / img.shape[1] > self.out_dim[0] / self.out_dim[1]:  # img more vertical than psf
            img = rescale(img, height=self.out_dim[0])
            res[:, (rw - img.shape[1]) // 2:(rw + img.shape[1]) // 2, :] = img
        else:  # img more horizontal than psf
            img = rescale(img, width=self.out_dim[1])
            res[(rh - img.shape[0]) // 2:(rh + img.shape[0]) // 2, :, :] = img
        return res.astype(np.float32) / 255

class SimForwardTrans:
    '''Simulates the forward model run on a normal image'''

    def __init__(self, psf_file='../../recon_files/psf_white_LED_Nick.tiff', rescale_fact=4):
        self.psf = imread_to_normalized_float(psf_file)
        self.psf = rescale(self.psf, height=self.psf.shape[0] // rescale_fact)

    def __call__(self, img):
        res = np.zeros(img.shape)
        for i in range(3):
            res[:, :, i] = fftconvolve(img[:, :, i], self.psf[:, :, i], mode='same')
            res[:, :, i] = res[:, :, i] / np.max(res[:, :, i])
        return (res*255).astype('uint8')


def rescale(img, width=None, height=None):
    if not width and not height:
        raise AssertionError
    if width:
        scale = width/img.shape[1]
        height = int(img.shape[0] * scale)
    else:
        scale = height / img.shape[0]
        width = int(img.shape[1] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


"""Reads image to float values between 0 and 1"""
def imread_to_normalized_float(im_name):
    return cv2.imread(im_name)[...,::-1].astype(np.float32)/255

if __name__ == '__main__':
    main()
