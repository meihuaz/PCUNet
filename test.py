import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.vis_utils import plot_single_pcd
from utils.train_utils import *
from pcn_data_loader import PCNDatasetTest
import numpy as np
from models import *
import random


def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


set_seed(1)


def test(net, dataloader_test, classx):
    metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    idx_to_plot = [i for i in range(0, 1600, 75)]

    logging.info('Testing...')
    if args.save_vis:
        save_gt_path = os.path.join(log_dir, 'pics', classx, 'gt')
        save_partial_path = os.path.join(log_dir, 'pics', classx, 'partial')
        save_completion_path = os.path.join(log_dir, 'pics', classx, 'completion')
        os.makedirs(save_gt_path, exist_ok=True)
        os.makedirs(save_partial_path, exist_ok=True)
        os.makedirs(save_completion_path, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):

            inputs_cpu, gt_cpu, _ = data

            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            result_dict = net(inputs, gt, is_training=False)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            cd_p = test_loss_meters['cd_p'].avg
            cd_t = test_loss_meters['cd_t'].avg
            f1 = test_loss_meters['f1'].avg

            # if i % args.step_interval_to_print == 0:
            #     logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                for j in range(args.batch_size):
                    idx = i * args.batch_size + j
                    if idx in idx_to_plot:
                        pic = 'object_%d.png' % idx
                        plot_single_pcd(result_dict['out2'][j].cpu().numpy(), os.path.join(save_completion_path, pic))
                        plot_single_pcd(gt_cpu[j], os.path.join(save_gt_path, pic))
                        plot_single_pcd(inputs_cpu[j].cpu().numpy(), os.path.join(save_partial_path, pic))

    return cd_p, cd_t, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('--config', help='path to config file', default='cfgs/atlasnet.yaml')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    average_cd_p_list = []
    average_cd_t_list = []
    average_f1_list = []

    classes = ['Plane', 'Cabinet', 'Car', 'Chair', 'Lamp', 'Couch', 'Table', 'Watercraft']
    for classx in classes:
        DATA_PATH = "/root/shenzhen_1_1/zmh/dataset/point_cloud/PCN_dataset/"
        dataset_test = PCNDatasetTest(root=DATA_PATH, input_size=2048, gt_size=args.num_points, classchoice=classx)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=int(args.workers), drop_last=False)
        dataset_length = len(dataset_test)
        print('*' * 100)
        print('class_choice:', classx, '  length:', dataset_length)

        cd_p, cd_t, f1 = test(net, dataloader_test, classx)
        cd_t = cd_t * 1e4

        print('average_cd_p: %4f' % cd_p, 'average_cd_t: %.4f' % cd_t,
              'average_f1: %.4f' % f1)

        average_cd_p_list.append(cd_p)
        average_cd_t_list.append(cd_t)
        average_f1_list.append(f1)

    average_cd_p_all_calsses = np.sum(average_cd_p_list) / len(average_cd_p_list)
    average_cd_t_all_calsses = np.sum(average_cd_t_list) / len(average_cd_t_list)
    average_f1_all_calsses = np.sum(average_f1_list) / len(average_f1_list)

    print('*' * 100)
    print('average_cd_p_all_calsses: %4f' % average_cd_p_all_calsses,
          'average_cd_t_all_calsses: %.4f' % average_cd_t_all_calsses,
          'average_f1_all_calsses: %.4f' % average_f1_all_calsses)
