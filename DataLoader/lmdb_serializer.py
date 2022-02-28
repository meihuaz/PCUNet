import numpy as np
import os
import open3d
import sys

from tqdm import tqdm
from tensorpack import dataflow


def main(lmdb_file_path, output_folder):

    lmdb_file_path = lmdb_file_path
    output_base_folder = output_folder

    df = dataflow.LMDBSerializer.load(lmdb_file_path, shuffle=False)
    df.reset_state()
    for d in tqdm(df, leave=False):
        ids = d[0].split('_')
        category_id = ids[0]
        model_id = ids[1]
        idx = len(ids) - 3

        # print(idx)

        partial_output_folder = os.path.join(output_base_folder, 'partial', category_id, model_id)
        complete_output_folder = os.path.join(output_base_folder, 'complete', category_id)
        if not os.path.exists(partial_output_folder):
            os.makedirs(partial_output_folder)
        if not os.path.exists(complete_output_folder):
            os.makedirs(complete_output_folder)

        p = open3d.geometry.PointCloud()
        p.points = open3d.utility.Vector3dVector(d[1].astype(np.float32))
        open3d.io.write_point_cloud(os.path.join(partial_output_folder, '%02d.pcd' % idx), p)
        p.points = open3d.utility.Vector3dVector(d[2].astype(np.float32))
        open3d.io.write_point_cloud(os.path.join(complete_output_folder, '%s.pcd' % model_id), p)


if __name__ == '__main__':
    lmdb_file_path = "/root/shenzhen/zmh/point_cloud/dataset/pcn_dataset/train.lmdb"
    output_base_folder = "/root/shenzhen/zmh/point_cloud/dataset/pcn_dataset/train2"
    main(lmdb_file_path, output_base_folder)
