from pix2pix.models.networks import define_D, define_G
import logging as log
from pix2pix.models.pix2pix_model import Pix2PixModel 
from pix2pix.options.train_options import TrainOptions
from pix2pix.util.visualizer import Visualizer
from pix2pix_wrapper import pix2pix_wrapped as Pix2Pix
from dataloader import NormalizedFaceDataset as Dataset
from torch.utils.data.dataloader import DataLoader
import time

pix2pix = Pix2Pix()
opt = pix2pix.opt
dataset = Dataset('/home/toby/Documents/HollowFakes/data/HF/cropped_boris')
loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)
visualizer = Visualizer(pix2pix.opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        loader_iter = iter(loader)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(loader_iter):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            pix2pix.model.set_input(data)
            pix2pix.model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(pix2pix.model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = pix2pix.model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                pix2pix.model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            pix2pix.model.save_networks('latest')
            pix2pix.model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        pix2pix.model.update_learning_rate()