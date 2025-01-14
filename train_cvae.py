import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import zlib
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn_limit, collation_fn_voxelmean, collation_fn_voxelmean_tta
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant

from util.nuscenes import nuScenes
from util.semantic_kitti_scene_flow import SemanticKITTI
from util.waymo import Waymo

from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv
from model.cvae_model import VAE

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_stratified_transformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # get model
    if args.arch == 'unet_spherical_transformer':
        from model.unet_spherical_transformer import Semantic as Model
        
        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(input_c=args.input_c, 
            m=args.m,
            classes=args.classes, 
            block_reps=args.block_reps, 
            block_residual=args.block_residual, 
            layers=args.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / args.quant_size_scale, 
            quant_size_sphere=window_size_sphere / args.quant_size_scale, 
            rel_query=args.rel_query, 
            rel_key=args.rel_key, 
            rel_value=args.rel_value, 
            drop_path_rate=args.drop_path_rate, 
            window_size_scale=args.window_size_scale, 
            grad_checkpoint_layers=args.grad_checkpoint_layers, 
            sphere_layers=args.sphere_layers,
            a=args.a,
        )
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    
    if args.condition_scene_flow:
        cvae_model = VAE([32, 16], 16, [16, 32], conditional=True, num_labels=args.classes, have_scene_flow=True)
    else:
        cvae_model = VAE([32, 16], 16, [16, 32], conditional=True, num_labels=args.classes)
    
    # set optimizer
    param_dicts = [
        {
            "params": cvae_model.parameters(),
            "lr": args.base_lr,
            "weight_decay": args.weight_decay,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        cvae_model = cvae_model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        cvae_model = torch.nn.parallel.DistributedDataParallel(cvae_model, device_ids=[gpu])
    else:
        # model = torch.nn.DataParallel(model.cuda())
        model = model.cuda()
        cvae_model = cvae_model.cuda()

    if main_process():
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        logger.info(cvae_model)
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    # set loss func 
    class_weight = args.get("class_weight", None)
    class_weight = torch.tensor(class_weight).cuda() if class_weight is not None else None
    if main_process():
        logger.info("class_weight: {}".format(class_weight))
        logger.info("loss_name: {}".format(args.get("loss_name", "ce_loss")))
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=args.ignore_label, reduction='none' if args.loss_name == 'focal_loss' else 'mean').cuda()
    
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 'nuscenes':
        train_data = nuScenes(args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_train.pkl'], 
            voxel_size=args.voxel_size, 
            split='train',
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            ignore_label=args.ignore_label,
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    
    elif args.data_name == 'semantic_kitti':
        train_data = SemanticKITTI(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95,1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
            for_cvae=True
        )
        # temp = train_data[0]

    elif args.data_name == 'waymo':
        train_data = Waymo(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95, 1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    collate_fn = partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None)
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True, 
        collate_fn=collate_fn
    )

    val_transform = None
    args.use_tta = getattr(args, "use_tta", False)
    if args.data_name == 'nuscenes':
        val_data = nuScenes(data_path=args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_val.pkl'], 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None),
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    elif args.data_name == 'semantic_kitti':
        val_data = SemanticKITTI(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
        # temp = val_data[0]
        # temp = val_data[1]
    elif args.data_name == 'waymo':
        val_data = Waymo(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("val_data samples: '{}'".format(len(val_data)))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    else:
        val_sampler = None
        
    if getattr(args, "use_tta", False):
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean_tta
        )
    else:
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean
        )

    # set scheduler
    if args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.val:
        validate(val_loader, model, criterion)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))
        
        if args.condition_scene_flow:
            train(train_loader, model, cvae_model, criterion, optimizer, epoch, scaler, scheduler, gpu, True)
        else:
            train(train_loader, model, cvae_model, criterion, optimizer, epoch, scaler, scheduler, gpu)
        
        if main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/cvae_model.pth'
            logger.info('Saving checkpoint to: ' + filename)
            if args.distributed:
                torch.save(cvae_model.module.state_dict(), filename)
            else:
                torch.save(cvae_model.state_dict(), filename)

    if main_process():
        writer.close()
        logger.info('==>Training done!')


def focal_loss(output, target, class_weight, ignore_label, gamma, need_softmax=True, eps=1e-8):
    mask = (target != ignore_label)
    output_valid = output[mask]
    if need_softmax:
        output_valid = F.softmax(output_valid, -1)
    target_valid = target[mask]
    p_t = output_valid[torch.arange(output_valid.shape[0], device=target_valid.device), target_valid] #[N, ]
    class_weight_per_sample = class_weight[target_valid]
    focal_weight_per_sample = (1.0 - p_t) ** gamma
    loss = -(class_weight_per_sample * focal_weight_per_sample * torch.log(p_t + eps)).sum() / (class_weight_per_sample.sum() + eps)
    return loss


def cvae_loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x)
        # MSE = nn.functional.mse_loss(recon_x, x)
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        # print(f'bce loss={BCE}, kld loss={KLD}')

        return BCE + KLD


def train(train_loader, model, cvae_model, criterion, optimizer, epoch, scaler, scheduler, gpu, condition_scene_flow=False):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    model.eval()
    for i, batch_data in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)

        data_time.update(time.time() - end)
        torch.cuda.empty_cache()
        
        coord, xyz, feat, target, offset = batch_data
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

        assert batch.shape[0] == feat.shape[0]
        
        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            
            with torch.no_grad():
                feat_model, output = model(sinput, xyz, batch, return_feat=True)   # output.shape = (n, num_class)
            assert output.shape[1] == args.classes

            # 将label为255的过滤掉
            valid_mask = ((target != 255) & (target == output.argmax(dim=-1)))
            feat_model, target = feat_model[valid_mask], target[valid_mask]
            
            # 将特征标准化到[0, 1]
            feat_model_min, feat_model_max = feat_model.min(), feat_model.max()
            feat_model = (feat_model - feat_model_min) / (feat_model_max - feat_model_min)
            
            # 取出对应的scene flow
            if condition_scene_flow:
                scene_flow = feat[:, 4:][valid_mask]
            
            if condition_scene_flow:
                recon_feat, mean, log_var, _ = cvae_model(feat_model, target, scene_flow)   # scene flow也作为条件
            else:
                recon_feat, mean, log_var, _ = cvae_model(feat_model, target)
            loss = cvae_loss_fn(recon_feat, feat_model, mean, log_var)

        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # if args.scheduler_update == 'step':
        #     scheduler.step()

        # 下面都是算指标的
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time, data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter,
                                                        lr=lr))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        (coord, xyz, feat, target, offset, inds_reconstruct) = batch_data
        inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reconstruct, :]
        
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
