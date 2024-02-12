import os
import random
import glob
import numpy as np
import torch
import yaml
import pickle
import sys
sys.path.append('/mnt/Disk16T/chenhr/SphereFormer')
from util.data_util import data_prepare
import scipy
from util.laser_mix_inst import lasermix_aug
from util.polar_mix_inst import polarmix
from util.instance_augmentation import instance_augmentation
from pathlib import Path


# used for polarmix
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]


#Elastic distortion
def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3
    bb = (np.abs(x).max(0)//gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x + g(x) * mag


def get_kitti_points_ringID(points):
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    yaw = -np.arctan2(scan_y, -scan_x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
    proj_y = np.zeros_like(proj_x)
    proj_y[new_raw] = 1
    ringID = np.cumsum(proj_y)
    ringID = np.clip(ringID, 0, 63)
    return ringID


class SemanticKITTI(torch.utils.data.Dataset):
    def __init__(self, 
        data_path, 
        voxel_size=[0.1, 0.1, 0.1], 
        split='train', 
        return_ref=True, 
        label_mapping="util/semantic-kitti.yaml", 
        rotate_aug=True, 
        flip_aug=True, 
        scale_aug=True, 
        scale_params=[0.95, 1.05], 
        transform_aug=True, 
        trans_std=[0.1, 0.1, 0.1],
        elastic_aug=False, 
        elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
        ignore_label=255, 
        voxel_max=None, 
        xyz_norm=False, 
        pc_range=None, 
        use_tta=None,
        vote_num=4,
        use_cross_da=False,
        for_cvae=False,
        for_finetune=False,
        instance_aug=False
    ):
        super().__init__()
        self.num_classes = 19
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.return_ref = return_ref
        self.split = split
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.scale_params = scale_params
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.data_path = data_path
        self.elastic_aug = elastic_aug
        self.elastic_gran, self.elastic_mag = elastic_params[0], elastic_params[1]
        self.use_tta = use_tta
        self.vote_num = vote_num
        self.use_cross_da = use_cross_da
        self.for_cvae = for_cvae
        self.for_finetune = for_finetune
        self.instance_aug = instance_aug

        if split == 'train':
            splits = semkittiyaml['split']['train']
        elif split == 'val':
            splits = semkittiyaml['split']['valid']
        elif split == 'test':
            splits = semkittiyaml['split']['test']
        elif split == 'trainval':
            splits = semkittiyaml['split']['train'] + semkittiyaml['split']['valid']
        else:
            raise Exception('Split must be train/val/test')

        self.files = []
        for i_folder in splits:
            self.files += sorted(glob.glob(os.path.join(data_path, "sequences", str(i_folder).zfill(2), 'velodyne', "*.bin")))
        
        if self.for_cvae:
            self.files = list(filter(lambda x: '000000' not in x, self.files))
        
        self.files_another = self.files.copy()
        random.shuffle(self.files_another)

        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)
        self.voxel_size = voxel_size
        
        # get class distribution weight 
        epsilon_w = 0.001
        origin_class = semkittiyaml['content'].keys()
        weights = np.zeros((len(semkittiyaml['learning_map_inv'])-1,),dtype = np.float32)
        for class_num in origin_class:
            if semkittiyaml['learning_map'][class_num] != 0:
                weights[semkittiyaml['learning_map'][class_num]-1] += semkittiyaml['content'][class_num]
        self.CLS_LOSS_WEIGHT = 1/(weights + epsilon_w)
        
        if self.instance_aug:
            self.inst_aug = instance_augmentation(data_path + '/instance_path.pkl', instance_classes, self.CLS_LOSS_WEIGHT,
                                                  random_flip=True, random_add=True, random_rotate=True, local_transformation=True)

    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.nusc_infos)
        return len(self.files)

    def __getitem__(self, index):
        if self.use_tta:
            samples = []
            for i in range(self.vote_num):
                sample = tuple(self.get_single_sample(index, vote_idx=i))
                samples.append(sample)
            return tuple(samples)
        return self.get_single_sample(index)

    def get_single_sample(self, index, vote_idx=0):

        file_path = self.files[index]

        raw_data = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
        if self.split != 'test':
            annotated_data = np.fromfile(file_path.replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.uint32).reshape((-1, 1))
            inst_data = annotated_data.copy()
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
        else:
            annotated_data = np.zeros((raw_data.shape[0], 1)).astype(np.int64)
        
        # load scene flow
        try:
            flow_data = np.load(file_path.replace('velodyne', 'flow')[:-3] + 'npy')
        except:
            flow_data = np.zeros((len(raw_data), 3), dtype=np.float32)
        raw_data = np.concatenate((raw_data, flow_data), axis=1)   # (n, 7)
            
        # laser mix and polar mix
        if self.use_cross_da:
            file_path_2 = self.files_another[index]
            raw_data_2 = np.fromfile(file_path_2, dtype=np.float32).reshape((-1, 4))
            if self.split != 'test':
                annotated_data_2 = np.fromfile(file_path_2.replace('velodyne', 'labels')[:-3] + 'label',
                                        dtype=np.uint32).reshape((-1, 1))
                inst_data_2 = annotated_data_2.copy()
                annotated_data_2 = annotated_data_2 & 0xFFFF  # delete high 16 digits binary
                annotated_data_2 = np.vectorize(self.learning_map.__getitem__)(annotated_data_2)
            else:
                annotated_data_2 = np.zeros((raw_data_2.shape[0], 1)).astype(np.int64)

            try:
                flow_data_2 = np.load(file_path_2.replace('velodyne', 'flow')[:-3] + 'npy')
            except:
                flow_data_2 = np.zeros((len(raw_data_2), 3), dtype=np.float32)
            raw_data_2 = np.concatenate((raw_data_2, flow_data_2), axis=1)
            assert len(annotated_data_2) == len(raw_data_2)   # (n_2, 7)
            
            prob = np.random.choice(2, 1)
            if prob == 1:   # laser mix
                raw_data, annotated_data, inst_data = lasermix_aug(
                    raw_data,
                    annotated_data,
                    inst_data,
                    raw_data_2,
                    annotated_data_2,
                    inst_data_2
                )
            elif prob == 0:
                alpha = (np.random.random() - 1) * np.pi
                beta = alpha + np.pi
                annotated_data = annotated_data.reshape(-1)
                annotated_data_2 = annotated_data_2.reshape(-1)
                inst_data = inst_data.reshape(-1)
                inst_data_2 = inst_data_2.reshape(-1)
                raw_data, annotated_data, inst_data = polarmix(
                    raw_data, annotated_data, inst_data, raw_data_2, annotated_data_2, inst_data_2,
                    alpha=alpha, beta=beta,
                    instance_classes=instance_classes, Omega=Omega
                )
                annotated_data = annotated_data.reshape(-1, 1)
                inst_data = inst_data.reshape(-1, 1)
        
        points = raw_data

        # Augmentation
        # ==================================================
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)   # 只对xy做旋转

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            if self.use_tta:
                flip_type = vote_idx % 4
            else:
                flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]

        if self.scale_aug:   # 只对xy做scale
            noise_scale = np.random.uniform(self.scale_params[0], self.scale_params[1])
            points[:, 0] = noise_scale * points[:, 0]
            points[:, 1] = noise_scale * points[:, 1]
            
        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            points[:, 0:3] += noise_translate
        
        # instance aug
        if self.instance_aug:
            inst_data = inst_data.astype(np.uint32)
            xyz, annotated_data, _, feat = self.inst_aug.instance_aug(points[:, 0:3], annotated_data.squeeze(), inst_data.squeeze(), points[:, 3:])
            points = np.concatenate((xyz, feat), axis=1)
            
        if self.elastic_aug:
            points[:, 0:3] = elastic(points[:, 0:3], self.elastic_gran[0], self.elastic_mag[0])
            points[:, 0:3] = elastic(points[:, 0:3], self.elastic_gran[1], self.elastic_mag[1])
        
        # random drop scene flow
        if (self.split == 'train' or self.split == 'trainval') and (np.random.rand() < 0.2) and (not self.for_cvae) and (not self.for_finetune):
            points[:, -3:] = 0
        # ==================================================
        
        if self.split != 'test':
            annotated_data[annotated_data == 0] = self.ignore_label + 1
            annotated_data = annotated_data - 1
            labels_in = annotated_data.astype(np.uint8).reshape(-1)
        else:
            labels_in = np.zeros(points.shape[0]).astype(np.uint8)

        feats = points   # (n', 7)
        xyz = points[:, :3]

        if self.pc_range is not None:
            xyz = np.clip(xyz, self.pc_range[0], self.pc_range[1])

        if self.split == 'train' or self.split == 'trainval':
            coords, xyz, feats, labels = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
            return coords, xyz, feats, labels
        else:
            coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
            if self.split == 'val':
                return coords, xyz, feats, labels, inds_reconstruct
            elif self.split == 'test':
                return coords, xyz, feats, labels, inds_reconstruct, self.files[index]


if __name__ == '__main__':
    train_data = SemanticKITTI('/mnt/Disk16T/chenhr/semantic_kitti', 
        voxel_size=[0.05, 0.05, 0.05], 
        split='train', 
        return_ref=True, 
        label_mapping='util/semantic-kitti.yaml', 
        rotate_aug=True, 
        flip_aug=True, 
        scale_aug=True, 
        scale_params=[0.95,1.05], 
        transform_aug=True, 
        trans_std=[0.1, 0.1, 0.1],
        elastic_aug=False, 
        elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
        ignore_label=255, 
        voxel_max=120000, 
        xyz_norm=False,
        pc_range=[[-51.2, -51.2, -4], [51.2, 51.2, 2.4]], 
        use_tta=False,
        use_cross_da=True,
        instance_aug=True
    )
    
    save_dir = Path('test/no_inst_aug')
    for i in range(len(train_data)):
        if i % 10 == 0:
            _, xyz, feats, _ = train_data[i]
            save_path = save_dir / f'{i}.txt'
            np.savetxt(save_path, xyz)
            
        if i == 100:
            break
        