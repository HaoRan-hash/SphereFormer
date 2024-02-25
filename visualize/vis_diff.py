from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm


gt_dir = Path('/mnt/Disk16T/chenhr/semantic_kitti/sequences/08/velodyne')
pred_dir = Path('/mnt/Disk16T/chenhr/SphereFormer/kitti_infer_val/sequences/08/predictions')

with open('util/semantic-kitti.yaml', 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
label_list = list(semkittiyaml['learning_map_inv'].values())[1:]

gt_files = sorted(list(gt_dir.iterdir()))
# print(gt_files)

for gt_file in tqdm(gt_files):
    raw_data = np.fromfile(gt_file, dtype=np.float32).reshape((-1, 4))[:, 0:3]
    diff = np.zeros((len(raw_data), 6), dtype=np.float32)
    diff[:, 0:3] = raw_data
    diff[:, 3:] = [158, 158, 159]
    
    annotated_file = str(gt_file).replace('velodyne', 'labels')[:-3] + 'label'
    annotated_data = np.fromfile(annotated_file, dtype=np.uint32).reshape((-1,))
    annotated_data = annotated_data & 0xFFFF
    pred_data = np.fromfile(pred_dir / Path(annotated_file).name, dtype=np.uint32).reshape((-1,))
    pred_data = pred_data & 0xFFFF
    
    mask = (annotated_data != pred_data)
    for i in label_list:
        temp = (annotated_data == i)
        temp_mask = (mask & temp)
        diff[temp_mask, 3:] = [0, 0, 0]
    
    np.savetxt(Path('visualize/difference') / (gt_file.stem + '.txt'), diff, fmt='%.4f')
