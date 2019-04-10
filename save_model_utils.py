import torch
import time
import numpy as np
import scipy
from PerceptualSimilarity.models import dist_model as dm


def preplot(image):
    image_color = np.zeros_like(image);
    image_color[:,:,0] = image[:,:,2]; image_color[:,:,1]  = image[:,:,1]
    image_color[:,:,2] = image[:,:,0];
    out_image = np.flipud(np.clip(image_color, 0,1))
    return out_image


def save_model_summary(model, test_loader, args):
   device = torch.cuda.current_device()
   model = model.to(device)

   loss_dict = test_training_images(model, test_loader, device)
   time_gpu, time_cpu = run_timing_test(model, test_loader, device)

   loss_dict['time_gpu'] = time_gpu
   loss_dict['time_cpu'] = time_cpu

   loss_dict['filename'] = 'net_' + args.net + '_ADMM_' + args.n_iters + '_dset_size_'\
                                      + args.dset_size + '_loss_' + args.loss_fn
   loss_dict['description'] = 'ADMM, muand tau,' +  args.n_iters + ' iterations, tau init 0.0002 ' + \
                               args.num_epochs + ' epochs, ' + args.dataset_len + ' dataset ' + args.dset_size

   save_filename = ('../saved_models/saved_stats/'+loss_dict['filename'])

   print('\r', 'Saving as:', save_filename, end = '')
   scipy.io.savemat(save_filename, loss_dict)
   return

def test_training_images(model, test_loader, device):
   model = model.eval()

   lpipsloss = dm.DistModel()
   lpipsloss.initialize(model='net-lin',net='alex',use_gpu=True,version='0.1')


   mse_loss = torch.nn.MSELoss(size_average=None)

   loss_dict = {'mse': [], 'mse_avg': 0,
                'psnr':[], 'psnr_avg': 0,
                'lpips': [], 'lpips_avg':0,
               }

   with torch.no_grad():
       for i_batch, sample_batched in enumerate(test_loader):
           print('\r', 'running test images, image:', i_batch, end = '')
           inputs = sample_batched['image'].to(device);
           labels = sample_batched['label'].to(device);

           output = model(inputs)

           mse_batch = mse_loss(output, labels)
           lpips_batch = lpipsloss.forward_pair(output, inputs)
           psnr_batch = 20 * torch.log10(1 / torch.sqrt(mse_batch))

           loss_dict['mse'].append(mse_batch.cpu().detach().numpy().squeeze())
           loss_dict['lpips'].append(lpips_batch.cpu().detach().numpy().squeeze())
           #loss_dict['lpips'].append(lpips_batch)
           loss_dict['psnr'].append(psnr_batch.cpu().detach().numpy().squeeze())

           if i_batch == 63:
               loss_dict['sample_image'] = preplot(output.detach().cpu().numpy()[0])

       print([l.shape for l in loss_dict['lpips']])
       loss_dict['mse_avg'] = np.average(loss_dict['mse']).squeeze()
       loss_dict['psnr_avg'] = np.average(loss_dict['psnr']).squeeze()
       loss_dict['lpips_avg'] = np.average(loss_dict['lpips'][:-1]).squeeze()
       print('\r', 'avg mse:', loss_dict['mse_avg'], 'avg psnr:', loss_dict['psnr_avg'], 'avg lpips:', loss_dict['lpips_avg'])


       return loss_dict

def run_timing_test(model, test_loader, device, num_trials = 100):
   print('\r', 'running timing test', end = '')
   t_avg_gpu = 0
   t_avg_cpu = 0

   with torch.no_grad():
       for i_batch, sample_batched in enumerate(test_loader):
           inputs = sample_batched['image'].to(device);
           break

   model = model.to(device)

   print('\r', 'running GPU timing test', end = '')
   for i in range(0,num_trials):
       t = time.time()
       output = model(inputs)
       elapsed = time.time() - t
       t_avg_gpu = t_avg_gpu + elapsed
       print(i)
   model_cpu = model.to('cpu')
   inputs_cpu = inputs.to('cpu')
   if model.__class__.__name__ == 'MyEnsemble':
      model_cpu.admm_model.to('cpu')
      model_cpu.denoise_model.to('cpu')

   print('\r', 'running CPU timing test', end = '')
   for i in range(0,num_trials):
      t = time.time()
      output_cpu = model_cpu(inputs_cpu)
      elapsed = time.time() - t
      t_avg_cpu = t_avg_cpu + elapsed


   t_avg_gpu = t_avg_gpu/num_trials
   t_avg_cpu = t_avg_cpu/num_trials

   return t_avg_gpu, t_avg_cpu