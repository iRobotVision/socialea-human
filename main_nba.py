import os
import random
import argparse

from csv import writer

import torch
import numpy as np

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mhtraj import MHTraj
from loaders.dataloader_nba import NBADataset

import matplotlib.pyplot as plt


def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)

    print('[INFO] The seed is:', seed)
    if not args.test:
        dataset_train = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode='train')
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=8, drop_last=True)
        
    dataset_test = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode='test' if args.test else 'val')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=8)

    model = MHTraj(opts).cuda()
    print(model)
    print('[INFO] Model params: {} M'.format(sum(p.numel() for p in model.parameters())/1000000))
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)

    if opts.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.decay_step, gamma=opts.decay_gamma)
    elif opts.scheduler_type == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=opts.decay_gamma)
    elif opts.scheduler_type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.num_epochs, eta_min=0.00005)

    model_save_dir = os.path.join('./checkpoints', os.path.basename(args.config).split('.')[0])
    os.makedirs(model_save_dir, exist_ok=True)

    if args.test:
        model_name = args.dataset + '_ckpt_best.pth'
        model_path = os.path.join(model_save_dir, model_name)
        print('[INFO] Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)
        ade, fde = test(model_ckpt['epoch'], model, loader_test)
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('./results', '{}_result.csv'.format(args.dataset)), 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow([os.path.basename(args.config).split('.')[0], ade, fde])
        exit()

    
    if args.vis:
        model_name = args.dataset + '_ckpt_best.pth'
        model_path = os.path.join(model_save_dir, model_name)
        print('[INFO] Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)
        vis(model_ckpt['epoch'], model, loader_test)

    results = {'epochs': [], 'losses': []}
    best_val_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    print('[INFO] The seed is :',seed)
    
    for epoch in range(0, opts.num_epochs):
        train(epoch, model, optimizer, loader_train)
        test_loss, ade = test(epoch, model, loader_test)
        results['epochs'].append(epoch)
        results['losses'].append(test_loss)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        # if test_loss < best_val_loss:  
        if ade < best_ade:
            best_val_loss = test_loss
            best_ade = ade
            best_epoch = epoch
            file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_best.pth')
            torch.save(state, file_path)
        print('[INFO] Best {} Loss: {:.5f} \t Best ade: {:.5f} \t Best epoch {}\n'.format(loader_test.dataset.mode.capitalize(), best_val_loss, best_ade, best_epoch))
        
        file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch) + '.pth')
        if epoch > 0:
            remove_file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch - 1) + '.pth')
            os.system('rm ' + remove_file_path)
            
        torch.save(state, file_path)
        
        if opts.scheduler_type is not None:
            scheduler.step()


def train(epoch, model, optimizer, loader):
    model.train()
    avg_meter = {'epoch': epoch, 'loss': 0, 'counter': 0}
    loader_len = len(loader)

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        
        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()        
        
        batch_size, num_agents, length, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]
        y_pred, pi = model(x_abs, x_rel)
        
        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred[..., :2] = torch.cumsum(y_pred[..., :2], dim=3) + cur_pos
            
        y = y[:, :, None, :, :]  # [B, N, 1, F, 2]
        
        total_loss = torch.mean(torch.min(torch.mean(torch.norm(y_pred[..., :2] - y, dim=-1), dim=3), dim=2)[0]) # for all agents
        # B, N, _, _, _ = y_pred.shape
        # l2_norm = torch.norm(y_pred[..., :2] - y, dim=-1)  # [B, N, K, F]
        # l2_norm = l2_norm.sum(-1)  # [B, N, K]
        # best_mode = l2_norm.argmin(dim=-1)  # [B, N]

        # batch_indices = torch.arange(B).unsqueeze(-1).expand(B, N)  # [B, N]
        # node_indices = torch.arange(N).unsqueeze(0).expand(B, N)  # [B, N]
        # y_pred_best = y_pred[batch_indices, node_indices, best_mode]  # [B, N, F, 4]
        
        # loc, scale = y_pred_best.chunk(2, dim=-1)
        # scale = scale.clone()
        # with torch.no_grad():
        #     scale.clamp_(min=1e-6)
        # nll = torch.log(2 * scale) + torch.abs(y.squeeze(2) - loc) / scale
        # reg_loss =  nll.mean()
        # # reg_loss =  F.l1_loss(y_pred_best, y.squeeze(2))
        # soft_target = F.softmax(-l2_norm / opts.future_length, dim=-1).detach()  # [B, N, K]
        # cls_loss = torch.sum(-soft_target * F.log_softmax(pi, dim=-1), dim=-1)
        # total_loss = reg_loss + cls_loss.mean()

        avg_meter['loss'] += total_loss.item() * batch_size * num_agents
        avg_meter['counter'] += (batch_size * num_agents)
        
        total_loss.backward()
        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
        optimizer.step()

        if i % 100 == 0:
            # th = get_th(opts, model)
            print('[{}][{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | LR: {}'
                  .format(args.dataset.upper(), 'TRAIN', epoch, opts.num_epochs - 1, i + 1, loader_len, total_loss.item(), optimizer.param_groups[0]['lr']))
    return avg_meter['loss'] / avg_meter['counter']


def test(epoch, model, loader):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade_1': 0, 'ade_2': 0, 'ade_3': 0, 'ade_4': 0, 'fde_1': 0, 'fde_2': 0, 'fde_3': 0, 'fde_4': 0, 'counter': 0}
    
    with torch.no_grad():
        for _, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()        
            
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]
            # x_rel[:, :, 0] = x_rel.new_zeros(1, 2)
            # x_len, x_heg = compute_angles_lengths_2D(x_rel)
            y_pred, _ = model(x_abs, x_rel)
            y_pred = y_pred[..., :2]
            # y_pred = model(x_abs, x_len, x_heg)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]
            
            ade_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :5] - y[:, :, :, :5], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 4:5] - y[:, :, :, 4:5], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            ade_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :10] - y[:, :, :, :10], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 9:10] - y[:, :, :, 9:10], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            ade_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :15] - y[:, :, :, :15], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 14:15] - y[:, :, :, 14:15], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            ade_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
                        
            avg_meter['ade_1'] += ade_1
            avg_meter['fde_1'] += fde_1
            avg_meter['ade_2'] += ade_2
            avg_meter['fde_2'] += fde_2
            avg_meter['ade_3'] += ade_3
            avg_meter['fde_3'] += fde_3
            avg_meter['ade_4'] += ade_4
            avg_meter['fde_4'] += fde_4
            
            avg_meter['counter'] += (num_agents * batch_size)
    
    # th = get_th(opts, model)
    print('\n[{}] Epoch {}'.format(loader.dataset.mode.upper(), epoch))
    print('[{}] minADE/minFDE (1.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_1'] / avg_meter['counter'], avg_meter['fde_1'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (2.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_2'] / avg_meter['counter'], avg_meter['fde_2'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (3.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_3'] / avg_meter['counter'], avg_meter['fde_3'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (4.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_4'] / avg_meter['counter'], avg_meter['fde_4'] / avg_meter['counter']))
    
    return avg_meter['fde_4'] / avg_meter['counter'], avg_meter['ade_4'] / avg_meter['counter']



def vis(epoch, model, loader):
    model.eval()
    count = 0
    with torch.no_grad():
        for _, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()        
            
            # batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]
            
            y_pred, _ = model(x_abs, x_rel)
            y_pred = y_pred[..., :2]

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            # y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            # y = np.array(y.cpu()) # B, N, T, 2
            # plot_preds(model, loader)
            # y = y[:, :, None, :, :]  # B, N, 1, T, 2

            past_traj = x_abs  # [B, N, H, 2]
            fut_traj = y  # [B, N, F, 2]
            traj = torch.cat((past_traj, fut_traj), dim=2) * 94 / 28  # [B, N, T, 2]
            traj = traj.cpu()
            # prediction = (y_pred - past_traj[:, :, -1:].unsqueeze(2))
            prediction = y_pred.view(-1, 20, 20, 2) * 94 / 28  # [B, N, K, F, 2]
            # prediction = prediction.transpose(1, 2)  # [B, K, N, F, 2]
            idx = [[0, 0], [2, 8], [3, 0], [4, 10]]
            
            for i in idx:
                scene = i[0]
                actor_num = i[1]

                plt.clf()

                ax = plt.axes(xlim=(Constant.X_MIN,
                                    Constant.X_MAX),
                            ylim=(Constant.Y_MIN,
                                    Constant.Y_MAX))
                ax.axis('off')
                fig = plt.gcf()
                ax.grid(False)  # Remove grid

                idx = scene*11 + actor_num

                colorteam1 = 'dodgerblue'
                colorteam2 = 'dodgerblue'
                colorball = 'dodgerblue'
                colorteam1_pre = 'skyblue'
                colorteam2_pre = 'skyblue'
                colorball_pre = 'skyblue'

                traj_pred = prediction[idx].cpu()  # [K, F, 2]
                # background players
                for actor_num_other in range(11):
                    zorder = 5
                    if actor_num_other==actor_num:
                        zorder = 105
                    traj_curr_ = traj[scene, actor_num_other].numpy()
                    if actor_num_other < 5:
                        color = colorteam1
                        color_pre = colorteam1_pre
                    elif actor_num_other < 10:
                        color = colorteam2
                        color_pre = colorteam2_pre
                    else:
                        color_pre = colorball_pre
                        color = colorball
                    for i in range(30):
                        points = [(traj_curr_[i,0],traj_curr_[i,1])]
                        (x, y) = zip(*points)
                        if i < 10:
                            plt.scatter(x, y, color=color_pre,s=15,alpha=1, zorder=zorder)
                        else:
                            if actor_num_other==actor_num and i==29:
                                plt.scatter(x, y, color=color,s=60, marker='*', alpha=1, zorder=zorder)

                            plt.scatter(x, y, color=color, s=15,alpha=1, zorder=zorder)
                    for i in range(29):
                        points = [(traj_curr_[i,0],traj_curr_[i,1]),(traj_curr_[i+1,0],traj_curr_[i+1,1])]
                        (x, y) = zip(*points)
                        if i < 10:
                            plt.plot(x, y, color=color_pre,alpha=0.5,linewidth=2, zorder=zorder-1)
                        else:
                            plt.plot(x, y, color=color,alpha=1,linewidth=2, zorder=zorder-1)
                

                for i in range(20):
                    plt.plot(traj_pred[i, :19, 0], traj_pred[i, :19, 1], 
                            linestyle='-', markersize=2, marker="|", alpha=1,
                            color='white',linewidth=1, zorder=101)
                    plt.plot(traj_pred[i,18:, 0], traj_pred[i, 18:, 1], 
                            linestyle='-', markersize=0, marker="o", alpha=1,
                            color='white',linewidth=1, zorder=101)
                    
                plt.scatter(traj_pred[:, -1, 0], traj_pred[:, -1, 1], 
                marker='*', 
                color='white',s=60, alpha=1, zorder=120)

                plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                        Constant.Y_MAX, Constant.Y_MIN],alpha=0.5)
                plt.imshow(mask, zorder=90, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                                    Constant.Y_MAX, Constant.Y_MIN],alpha=0.7)
                
                plt.savefig('/home/robot/Documents/cjy/Trajectory_Prediction/MART-main/visualization/nba_{}.png'.format(count))
                count += 1


            

court = plt.imread("/home/robot/Documents/cjy/Trajectory_Prediction/MART-main/visualization/nba/court.png")
mask = np.zeros_like(court)

class Constant:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 6
    X_MIN = 0
    X_MAX = 100
    Y_MIN = 0
    Y_MAX = 50
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='nba', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_nba.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="4", help='gpu id')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--vis", action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()