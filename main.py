import argparse
import os
import numpy as np
import torch

from model.utils import set_gpu, ensure_path, AverageMeter, Timer
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tensorboardX import SummaryWriter

import time
import datetime

import logging
import sys

import os.path as osp

#from models import vgg19

'''Train Benchmark'''

def get_args():
    parser = argparse.ArgumentParser()
    # Basic Parameters
    parser.add_argument('--dataset', type=str, default='SHHB',
        choices=['SHHA', 'SHHB', 'cdpeople', 'cdvehicle'])
    parser.add_argument('--backbone_class', type=str, default='vgg19',
        choices=['VGG16', 'Res12', 'Res18', 'resnet18', 'pretrain', 'vgg19']) 

    # Optimization Parameters
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=225)
    parser.add_argument('--prefetch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--lambda_reg', type=float, default=0.1)

    # for video
    parser.add_argument('--seg_len', type=int, default=3)

    # Model Parameters
    parser.add_argument('--model_type', type=str, default='SACANet',
        choices=['MCNN', 'AlexNet', 'VGG', 'VGG_DECODER', 'Res50', 'Res101', 'SACANet'])

    # Other Parameters
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--LOG_PARA', type=float, default=100)#100

    args, unknown_args = parser.parse_known_args()

    set_gpu(args.gpu)

    save_path1 = '{}-{}-{}'.format(args.dataset, args.model_type, args.backbone_class)
    save_path2 = '_'.join([str(args.lr), str(args.batch_size), str(args.max_epoch), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    args.init_path = os.path.join('./saves/initialization', 'resnet18.pth')
    args.pretrain_path = os.path.join('./saves')

    args.save_path1 = os.path.join('./exp_log/', save_path1)
    args.save_path2 = os.path.join('./exp_log/', save_path1, save_path2)

    create_exp_dir(args.save_path1)
    create_exp_dir(args.save_path2)
    return args

def get_model(args, s1_data):
    if args.model_type == 'SACANet':
        from model.models.SACANet import SACANet
        model = SACANet(args)

    else:
        raise ValueError('No Such Model')

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    return model



def get_loader(args):
    if args.dataset == 'SHHB':
        from dataloader.shhbLoader import shhbDataset as dataset
        trainset = dataset('train', args)
        valset = dataset('val', args)
        testset = dataset('test', args)
    elif args.dataset == 'SHHA':
        from dataloader.shhaLoader import shhaDataset as dataset
        trainset = dataset('train', args)
        valset = dataset('val', args)
        testset = dataset('test', args)
    elif args.dataset == 'cdpeople':
        from dataloader.cdpeopleLoader import cdpeopleDataset as dataset
        trainset = dataset('train', args)
        testset = dataset('val', args)
    elif args.dataset == 'cdvehicle':
        from dataloader.cdvehicleLoader import cdvehicleDataset as dataset
        trainset = dataset('train', args)
        testset = dataset('val', args)
    else:
        raise ValueError('Non-supported Dataset.')

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True) 
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)  

    return train_loader, test_loader, test_loader

def create_exp_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

def train_model(args, model, train_loader, val_loader, logging):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
    criterion = nn.MSELoss().cuda()
    reg1 = nn.L1Loss().cuda()
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path1, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_mae'] = []
    trlog['train_mse'] = []
    trlog['val_mae'] = []
    trlog['val_mse'] = []

    trlog['max_mae'] = 1000000
    trlog['max_mse'] = 1000000
    trlog['max_mae_epoch'] = 0

    trlog['max_mae_last10'] = 1000000
    trlog['max_mse_last10'] = 1000000
    trlog['max_mae_last10_epoch'] = 0

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(logdir=args.save_path1)
    epoch_time = AverageMeter()

    for epoch in range(1, args.max_epoch + 1):
        epoch_start = time.time()
        model.train()
        t1 = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        batch_time = AverageMeter()

        for i, batch in enumerate(train_loader, 1):
            batch_start = time.time()
            global_count = global_count + 1
            
            if args.model_type == 'SACANet':
                data, gt_label = batch[0].cuda(), batch[1].cuda()
                pred_map = model(data)
                pred_map = torch.squeeze(pred_map)
                gt_label = torch.squeeze(gt_label)
                loss = criterion(pred_map, gt_label)



            else:
                raise ValueError('')
             
            pred_map = pred_map.data.cpu().numpy()
            gt_label = gt_label.data.cpu().numpy()

            for i_img in range(pred_map.shape[0]):
                
                pred_cnt = np.sum(pred_map[i_img])/args.LOG_PARA
                
                gt_count = np.sum(gt_label[i_img])/args.LOG_PARA
                
                maes.update(abs(gt_count-pred_cnt))
                mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))

            writer.add_scalar('data/loss', float(loss), global_count)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            batch_duration = time.time() - batch_start
            batch_time.update(batch_duration)

            t1.update(loss.item(), data.size(0))
   
        t1 = t1.avg
        mae = maes.avg

        mse = np.sqrt(mses.avg)        

        lr_scheduler.step()

        epoch_duration = time.time() - epoch_start
        epoch_time.update(epoch_duration)
        logging.info('epoch {}, loss={:4f}, train mae={:4f}, train mse={:4f}'.format(epoch, float(t1), float(mae), float(mse)))
        logging.info('Epoch time: {:3f}s'.format(epoch_duration))

        v1 = AverageMeter()
        vmaes = AverageMeter()
        vmses = AverageMeter()
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
            
                if args.model_type == 'CSRNet':
                    data, gt_label = batch[0].cuda(), batch[1].cuda()
                    pred_map = model(data)
                    pred_map = torch.squeeze(pred_map)
                    gt_label = torch.squeeze(gt_label)
                    loss = criterion(pred_map, gt_label)

                else:
                    raise ValueError('')


                vmaes.update(abs(gt_count-pred_cnt))
                vmses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))


        v1 = v1.avg
        vmae = vmaes.avg
        vmse = np.sqrt(vmses.avg)

        writer.add_scalar('data/val_loss', float(v1), epoch)
        logging.info('epoch {}, val mae={:}, val mse={:}'.format(epoch, vmae, vmse))
       
        if epoch % 10 == 0 or epoch > (args.max_epoch-30):
            if vmae < trlog['max_mae']:
                trlog['max_mae'] = vmae
                trlog['max_mse'] = vmse
                trlog['max_mae_epoch'] = epoch
                save_model('max_acc')

            if epoch >= (args.max_epoch - 10):
                if vmae <= trlog['max_mae_last10']:
                    trlog['max_mae_last10'] = vmae
                    trlog['max_mse_last10'] = vmse
                    trlog['max_mae_last10_epoch'] = epoch
            
            trlog['train_loss'].append(t1)
            trlog['train_mae'].append(mae)
            trlog['train_mse'].append(mse)
            trlog['val_loss'].append(v1)
            trlog['val_mae'].append(vmae)
            trlog['val_mse'].append(vmse)

            torch.save(trlog, osp.join(args.save_path1, 'trlog'))



            logging.info('best epoch {}, best val mae={:.4f}, best val mse={:.4f}'.format(trlog['max_mae_epoch'], trlog['max_mae'], trlog['max_mse']))
            logging.info('best val mae last 10 epoch {}, val mae last10={}, val mse last10={:.4f}'.format(trlog['max_mae_last10_epoch'], trlog['max_mae_last10'], trlog['max_mse_last10']))
            logging.info('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

        logging.info('Total epoch training time: {:.3f}s, average: {:.3f}s'.format(epoch_time.sum, epoch_time.avg))

    writer.close()

    logging.info(args.save_path1)
    return model


def test_model(args, model, test_loader, logging):
    trlog = torch.load(osp.join(args.save_path1, 'trlog'))
    model.load_state_dict(torch.load(osp.join(args.save_path1, 'max_acc.pth'))['params'])

    t1 = AverageMeter()
    tmaes = AverageMeter()
    tmses = AverageMeter()
    model.eval()

    logging.info('Best Epoch {}, best val mae={:.4f}, best val mse={:.4f}'.format(trlog['max_mae_epoch'], trlog['max_mae'], trlog['max_mse']))
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):

            data1, data2, data3, gt_label1, gt_label2, gt_label3 = batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].cuda(),batch[4].cuda(),batch[5].cuda() 

            if args.model_type == 'SACANet':
                data, gt_label = batch[0].cuda(), batch[1].cuda()
                pred_map = model(data)
                loss = criterion(pred_map, gt_label)

            else:
                raise ValueError('')

            pred_map = pred_map[:,1,:,:].data.cpu().numpy()
            gt_label = gt_label[:,1,:,:].data.cpu().numpy()
            for i_img in range(pred_map.shape[0]):
                
                pred_cnt = np.sum(pred_map[i_img])/args.LOG_PARA
                gt_count = np.sum(gt_label[i_img])/args.LOG_PARA

                tmaes.update(abs(gt_count-pred_cnt))
                tmses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))

            t1.update(loss.item(), data.size(0))

    t1 = t1.avg
    tmae = tmaes.avg
    tmse = np.sqrt(tmses.avg)

    logging.info('Test mae={:.4f}, mse={:.4f}'.format(tmae, tmse))


if __name__ == '__main__':
    args = get_args()
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_path1, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(vars(args))

    train_loader, val_loader, test_loader = get_loader(args)

    model = get_model(args, logging)

    model = train_model(args, model, train_loader, val_loader, logging)



