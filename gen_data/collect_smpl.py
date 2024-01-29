"""
Will collect the pkl file of SMPL-X parameters and fuse them into an npz
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, export_dotdict, to_tensor, export_pts, export_mesh
from easyvolcap.utils.easy_utils import read_camera, write_camera, read_pickle

from easymocap.bodymodel.smpl import SMPLModel
from smplx.body_models import SMPL, SMPLX
import torch

@catch_throw
def main():
    args = dotdict()
    args.zjumocap_root = 'data/my_zju_mocap/my_313'
    args.smplx_root = '/home/xuzhen/code/smplx/output'
    args.frame_sample = [0, 120, 1]
    args.smpl_params_file = 'smpl_params.npz'
    args.keys = ['global_orient', 'transl', 'body_pose', 'jaw_pose', 'betas', 'expression', 'left_hand_pose', 'right_hand_pose']
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    smpl_params = dotdict()
    pkls = sorted(os.listdir(args.smplx_root))
    for pkl in pkls:
        pkl = read_pickle(join(args.smplx_root, pkl))
        for key in args.keys:
            if key not in smpl_params:
                smpl_params[key] = pkl[key].detach().cpu().numpy()[None]
            else:
                smpl_params[key] = np.concatenate([smpl_params[key], pkl[key].detach().cpu().numpy()[None]], axis=0)

    