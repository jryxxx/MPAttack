import glob
import os
import numpy as np


def loop_npz(folder):
    for npz_file in glob.glob(os.path.join(folder, "*.npz")):
        data = np.load(npz_file)
        veh_trans, cam_trans = data['veh_trans'], data['cam_trans']
        if veh_trans[1].any():
            print(npz_file)


if __name__ == "__main__":
    loop_npz("/root/autodl-tmp/dataset/val/npz")
    loop_npz("/root/autodl-tmp/data_tgrs2/test/npz")

        

