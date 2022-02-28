import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from DataLoader.pcn_data_loader import PCNDatasetTest, PCNDataset
import numpy as np
from utils.train_utils import *
import random
from utils.visu_util import plot_pcd_three_views
from torch.autograd import Variable


def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


set_seed(1)


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


def test(net, dataloader_test, classx):
    metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}

    ratio = 0.6
    num_points = 16384
    CROP_POINT_NUM = int(num_points * ratio)

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

            _, complete, _ = data

            # gt = complete

            # inputs = inputs_cpu.float().cuda()
            # gt = gt_cpu.float().cuda()

            complete = complete.float().cuda()
            bs = complete.shape[0]

            input_cropped = torch.FloatTensor(bs, args.num_points, 3).cuda()
            choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                      torch.Tensor([-1, 0, 0]),
                      torch.Tensor([-1, 1, 0])]

            for m in range(bs):
                index = random.sample(choice, 1)
                p_center = index[0]
                N, _ = complete[m].shape
                p_center = p_center.reshape(1, 3).repeat(N, 1).cuda()
                dist = (complete[m] - p_center)
                dist = torch.mul(dist, dist)
                dist = torch.sum(dist, dim=1)
                sorted, indices = torch.sort(dist)
                cropped = complete[m, indices[CROP_POINT_NUM:]]
                cropped = resample_pcd(cropped, 2048)
                input_cropped[m] = cropped
            inputs = Variable(input_cropped, requires_grad=True)
            inputs = inputs.transpose(2, 1).contiguous()


            # gt = complete
            gt = torch.FloatTensor(bs, args.num_points, 3).cuda()
            for m in range(bs):
                gt[m] = resample_pcd(complete[m], 2048)

            result_dict = net(inputs, gt, is_training=False)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            cd_p = result_dict['cd_p']
            cd_t = result_dict['cd_t']
            f1 = result_dict['f1']
            # print('i', i, 'cd_p', cd_p, 'cd_t', cd_t * 1e4, 'f1', f1)

            cd_p = test_loss_meters['cd_p'].avg
            cd_t = test_loss_meters['cd_t'].avg
            f1 = test_loss_meters['f1'].avg

            # if i % args.step_interval_to_print == 0:
            #     logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                # for j in range(arg.batch_size):
                idx = i * arg.batch_size
                pic = 'object_%d.pdf' % idx

                plot_pcd_three_views(filename=os.path.join(save_completion_path, pic.replace('.pdf', 'out3.pdf')),
                                     pcds=result_dict['out3'].cpu().numpy(), titles=str(result_dict['cd_t']))
                plot_pcd_three_views(filename=os.path.join(save_completion_path, pic.replace('.pdf', 'out2.pdf')),
                                     pcds=result_dict['out2'].cpu().numpy(), titles=str(result_dict['cd_t']))
                plot_pcd_three_views(filename=os.path.join(save_completion_path, pic.replace('.pdf', 'out1.pdf')),
                                     pcds=result_dict['out1'].cpu().numpy(), titles=str(result_dict['cd_t']))
                plot_pcd_three_views(filename=os.path.join(save_gt_path, pic), pcds=gt_cpu,
                                     titles=str(result_dict['cd_t']))
                plot_pcd_three_views(filename=os.path.join(save_partial_path, pic),
                                     pcds=inputs_cpu.cpu().numpy(),
                                     titles=str(result_dict['cd_t']))
                plot_pcd_three_views(filename=os.path.join(save_partial_path, pic.replace('.pdf', 'xyz1.pdf')),
                                     pcds=result_dict['input_xyz1'].cpu().numpy(),
                                     titles=str(result_dict['input_xyz1']))
                plot_pcd_three_views(filename=os.path.join(save_partial_path, pic.replace('.pdf', 'xyz2.pdf')),
                                     pcds=result_dict['input_xyz2'].cpu().numpy(),
                                     titles=str(result_dict['input_xyz2']))

    return cd_p, cd_t, f1


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('--config', help='path to config file', default='cfgs/cascade.yaml')
    parser.add_argument('--batch_size', help='path to config file', default=32)
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

    parameters = get_parameter_number(net)

    average_cd_p_list = []
    average_cd_t_list = []
    average_f1_list = []
    #
    classes = ['Plane', 'Cabinet', 'Car', 'Chair', 'Lamp', 'Couch', 'Table', 'Watercraft']
    for classx in classes:
        DATA_PATH = "/root/shenzhen_1_1/zmh/dataset/point_cloud/PCN_dataset/"
        dataset_test = PCNDatasetTest(root=DATA_PATH, input_size=2048, gt_size=16384, classchoice=classx)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=arg.batch_size,
                                                      shuffle=False, num_workers=int(args.workers))
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
