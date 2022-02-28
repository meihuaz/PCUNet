import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from DataLoader.pcn_data_loader import PCNDataset
import torch.optim as optim
import torch
from utils.train_utils import *
from models import *


def train():
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    DATA_PATH = "/root/shenzhen_1_1/zmh/dataset/point_cloud/PCN_dataset/"
    dataset = PCNDataset(root=DATA_PATH, input_size=2048, gt_size=args.num_points, split='train')
    dataset_test = PCNDataset(root=DATA_PATH, input_size=2048, gt_size=args.num_points, split='valid')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model())
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    # Create the optimizers
    betas = args.BETAS.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.WEIGHT_DECAY,
                                 betas=betas)
    # if args.load_model:
    #     ckpt = torch.load(args.load_model)
    #     net.module.load_state_dict(ckpt['net_state_dict'])
    #     if cascade_gan:
    #         net_d.module.load_state_dict(ckpt['D_state_dict'])
    #     logging.info("%s's previous weights loaded." % args.model_name)

    cur_epoch = args.start_epoch

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
        cur_epoch = ckpt['epoch']

    for epoch in range(cur_epoch, args.nepoch):
        train_loss_meter.reset()
        net.module.train()

        if args.lr_decay:
            lr = args.lr
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                for i, lr_decay_epoch in enumerate(decay_epoch_list):
                    if epoch >= lr_decay_epoch:
                        lr *= decay_rate_list[i]

            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            inputs, gt = data
            # mean_feature = None

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            # inputs = inputs.transpose(2, 1).contiguous()
            cd3, loss_cd, loss_pmd, net_loss = net(inputs, gt)

            train_loss_meter.update(net_loss.mean().item())
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, loss2: %f total_loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss,
                              cd3.mean().item(), net_loss.mean().item(), lr))

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")

            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(log_dir, str(epoch) + 'checkpoint.tar'))

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            inputs, gt = data
            # mean_feature = None

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            # inputs = inputs.transpose(2, 1).contiguous()
            # result_dict = net(inputs, gt, is_training=False, mean_feature=mean_feature)
            result_dict = net(inputs, gt, is_training=False)
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    # parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('--config', help='path to config file', default='cfgs/pmpnet.yaml')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    train()
