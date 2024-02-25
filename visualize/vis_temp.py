from pathlib import Path
import yaml
import numpy as np
import random


# 生成000050的真值和预测结果

# with open('util/semantic-kitti.yaml', 'r') as stream:
#     semkittiyaml = yaml.safe_load(stream)
# color_map = semkittiyaml['color_map']
# print(color_map)

# gt_file = Path('/mnt/Disk16T/chenhr/semantic_kitti/sequences/08/velodyne/000050.bin')
# pred_file = Path('/mnt/Disk16T/chenhr/SphereFormer/kitti_infer_val/sequences/08/predictions/000050.label')

# raw_data = np.fromfile(gt_file, dtype=np.float32).reshape((-1, 4))[:, 0:3]
# gt_save = np.zeros((len(raw_data), 6), dtype=np.float32)
# gt_save[:, 0:3] = raw_data
# pred_save = np.zeros((len(raw_data), 6), dtype=np.float32)
# pred_save[:, 0:3] = raw_data

# annotated_file = str(gt_file).replace('velodyne', 'labels')[:-3] + 'label'
# annotated_data = np.fromfile(annotated_file, dtype=np.uint32).reshape((-1,))
# annotated_data = annotated_data & 0xFFFF
# pred_data = np.fromfile(pred_file, dtype=np.uint32).reshape((-1,))
# pred_data = pred_data & 0xFFFF

# for i, j in color_map.items():
#     mask_1 = (annotated_data == i)
#     mask_2 = (pred_data == i)
    
#     gt_save[mask_1, 3:] = j
#     pred_save[mask_2, 3:] = j

# np.savetxt(Path('visualize') / (gt_file.stem + '_gt.txt'), gt_save, fmt='%.4f')
# np.savetxt(Path('visualize') / (pred_file.stem + '_pred.txt'), pred_save, fmt='%.4f')


# ============================================================================= #

# 生成000050的特征分数和掩码
# diff = np.loadtxt('visualize/difference/000050.txt', dtype=np.float32)

# mask_save = np.copy(diff)
# score_save = np.zeros((len(diff), 6), dtype=np.float32)
# score_save[:, 0:3] = diff[:, 0:3]

# temp_mask = (diff[:, -1] == 0)
# mask_save[temp_mask, 3:] = [255, 0, 0]
# np.savetxt('visualize/000050_mask.txt', mask_save, fmt='%.4f')

# deep_red = [[255, 0, 0], [255, 22, 22], [255, 45, 45]]
# shadow_red = [[255, 158, 158], [255, 181, 181], [255, 204, 204]]

# for i in range(len(diff)):
#     j = random.randint(0, 2)
#     if temp_mask[i]:
#         score_save[i, 3:] = deep_red[j]
#     else:
#         score_save[i, 3:] = shadow_red[j]
        
# np.savetxt('visualize/000050_score.txt', score_save, fmt='%.4f')

# =============================================================================== #

# 生成000050的低质过滤结果
# diff = np.loadtxt('visualize/difference/000050.txt', dtype=np.float32)
# temp_mask = (diff[:, -1] != 0)

# low_quality_fea = np.copy(diff)
# low_quality_fea = low_quality_fea[temp_mask]

# np.savetxt('visualize/000050_low_quality_fea.txt', low_quality_fea, fmt='%.4f')

# annotated_data = np.loadtxt('visualize/000050_gt.txt', dtype=np.float32)
# annotated_data = annotated_data[temp_mask]

# np.savetxt('visualize/000050_low_quality_gt.txt', annotated_data, fmt='%.4f')

# ============================================================================= # 

# 生成000050的场景流
flow_file = '/mnt/Disk16T/chenhr/semantic_kitti/sequences/08/flow/000050.npy'


