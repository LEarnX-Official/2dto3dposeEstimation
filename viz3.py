from __future__ import print_function, absolute_import, division

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import io
import datetime
import argparse
import numpy as np
import os.path as path
import torch
import imageio.v2 as imageio
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from network.GraFormer import GraFormer, adj_mx_from_edges
from tqdm import tqdm
from IPython.display import HTML
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True]]]).cuda()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='Directions', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs (default: 20)')

    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=10, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=1, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=50000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='gamma of learning rate decay')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    # Add this line to include max_norm argument
    parser.add_argument('--max_norm', default=1, type=float, help='max norm for gradient clipping')

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args



def visualize_predictions(predictions: np.ndarray, ground_truth: np.ndarray, actions: list, output_path: str = 'visualization.gif'):
    """Visualize 3D poses and display the animation inline in a Jupyter notebook."""

    # Define edges representing the connections between joints
    edges = [(0, 1), (1, 2), (2, 3),
             (0, 4), (4, 5), (5, 6),
             (0, 7), (7, 8), (8, 9),
             (8, 10), (10, 11), (11, 12),
             (8, 13), (13, 14), (14, 15)]

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    def set_axes_equal(ax):
        """Set 3D plot axes to have equal scale."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        max_range = max(x_range, y_range, z_range)
        x_center = (x_limits[1] + x_limits[0]) / 2
        y_center = (y_limits[1] + y_limits[0]) / 2
        z_center = (z_limits[1] + z_limits[0]) / 2

        ax.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
        ax.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
        ax.set_zlim([z_center - max_range / 2, z_center + max_range / 2])

    def update(frame):
        ax.cla()  # Clear the axis
        ax.set_title(f'Frame: {frame}, Action: {actions[frame]}')
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Y (Front/Back)')
        ax.set_zlabel('Z (Up/Down)')

        # Plot ground truth
        gt = ground_truth[frame]
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='b', label='Ground Truth')
        for start, end in edges:
            ax.plot([gt[start, 0], gt[end, 0]], [gt[start, 1], gt[end, 1]], [gt[start, 2], gt[end, 2]], 'b')

        # Plot predictions
        pred = predictions[frame]
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', label='Prediction')
        for start, end in edges:
            ax.plot([pred[start, 0], pred[end, 0]], [pred[start, 1], pred[end, 1]], [pred[start, 2], pred[end, 2]], 'r')

        # Ensure equal scaling
        set_axes_equal(ax)
        ax.legend(loc='upper right')

        # Adjust view to show skeleton from a conventional perspective
        ax.view_init(elev=30, azim=210)  # Adjust view to a more intuitive angle

    # Check for valid number of frames
    if len(predictions) == 0 or len(ground_truth) == 0:
        raise ValueError("Predictions or ground truth arrays are empty.")

    # Create an animation
    anim = FuncAnimation(fig, update, frames=min(500, len(predictions)), interval=100)

    # Save the animation to a file
    anim.save(output_path, writer='imagemagick', fps=60, dpi=80)
    print(f"Visualization saved to {output_path}")

    # Return HTML representation for display in a Jupyter notebook
    return HTML(anim.to_jshtml())

# Optionally increase the embed limit if neededa
plt.rcParams['animation.embed_limit'] = 100 * 1024 * 1024  # 100 MB limit

def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', 'test_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'train_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = list(map(lambda x: dataset.define_actions(x)[0], action_filter))  # Convert map to list
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    # Create model
    print("==> Creating model...")
    edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                         [0, 4], [4, 5], [5, 6],
                         [0, 7], [7, 8], [8, 9],
                         [8, 10], [10, 11], [11, 12],
                         [8, 13], [13, 14], [14, 15]], dtype=torch.long)
    adj = adj_mx_from_edges(num_pts=16, edges=edges, sparse=False)
    model_pos = GraFormer(adj=adj.cuda(), hid_dim=args.dim_model, coords_dim=(2, 3), n_pts=16,
                              num_layers=args.n_layer, n_head=args.n_head, dropout=args.dropout).cuda()

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)


    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        ckpt_dir_path = path.join(args.checkpoint, 'GTNet_V3_cheb_2l-' + args.keypoints,
                                  '_head-%s' % args.n_head + '-layers-%s' % args.n_layer + '-dim-%s' % args.dim_model,
                                  '_lr_step%s' % args.lr_decay + '-lr_gamma%s' % args.lr_gamma + '-drop_%s' % args.dropout)
        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))
        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])


    if args.evaluate:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, [action], stride)
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
            errors_p1[i], errors_p2[i] = evaluate(valid_loader, model_pos, device, action, args.checkpoint)

        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
        print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
        exit(0)


    poses_train, poses_train_2d, actions_train = fetch(subjects_train, dataset, keypoints, action_filter, stride)
    train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, actions_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, action_filter, stride)
    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now,
                                              glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)

        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device, 'Evaluation')

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))

    return


def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    dl_ = tqdm(data_loader)
    for i, (targets_3d, inputs_2d, _) in enumerate(dl_):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)

        outputs_3d = model_pos(inputs_2d, src_mask)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


def evaluate(data_loader, model_pos, device, action=None, checkpoint_dir=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))

    all_predictions = []
    all_ground_truth = []

    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d, src_mask).cpu()
        outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)

        all_predictions.append(outputs_3d.numpy())
        all_ground_truth.append(targets_3d.numpy())

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()

    # Convert lists to numpy arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # If action is provided, use it for visualization
    if action and checkpoint_dir:
        # Ensure action name is valid for filenames
        action_name = action.replace(' ', '_').replace('/', '_')  # Replace invalid filename characters
        output_path = path.join(checkpoint_dir, f'{action_name}_visualization.gif')
        visualize_predictions(all_predictions, all_ground_truth, actions=data_loader.dataset._actions, output_path=output_path)

    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg



if __name__ == '__main__':
    main(parse_args())
