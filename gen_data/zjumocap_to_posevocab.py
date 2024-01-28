"""
Convert the `motion.npz` easymocap pose to posevocab pose `smpl_params.npz`.
Convert the `extri.yml` and `intri.yml` easyvolcap camera parameters to posevocab camera parameters `calibration.json`.

motion.npz:
['poses', 'Rh', 'Th', 'shapes']

smpl_params.npz
['global_orient', 'transl', 'body_pose', 'jaw_pose', 'betas', 'expression', 'left_hand_pose', 'right_hand_pose']

Only using ['global_orient', 'transl', 'body_pose', 'betas'] for now

calibration.json
['K', 'R', 'T', 'distCoeff', 'imgSize', 'rectifyAlpha']
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, export_dotdict
from easyvolcap.utils.easy_utils import read_camera, write_camera


@catch_throw
def main():
    args = dotdict()
    args.zjumocap_root = 'data/my_zju_mocap/my_313'
    args.posevocab_root = 'data/my_zju_mocap/my_313'
    args.motion_file = 'motion.npz'
    args.smpl_params_file = 'smpl_params.npz'
    args.calibration_file = 'calibration.json'
    args.h_w_overwrite = [1024, 1024]
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    motion = load_dotdict(join(args.zjumocap_root, args.motion_file))
    smpl_params = dotdict()
    smpl_params.global_orient = motion.Rh
    smpl_params.transl = motion.Th
    smpl_params.body_pose = motion.poses[:, 3:72-6] # 21 * 3
    smpl_params.betas = motion.shapes[:, :10]
    export_dotdict(smpl_params, join(args.posevocab_root, args.smpl_params_file))

    cameras = read_camera(join(args.zjumocap_root))
    calibration = dotdict()
    for cam in cameras:
        calibration[cam] = dotdict()
        calibration[cam].K = cameras[cam].K.reshape(-1).tolist()
        calibration[cam].R = cameras[cam].R.tolist()
        calibration[cam].T = cameras[cam].T.reshape(-1).tolist()
        calibration[cam].distCoeff = cameras[cam].D.reshape(-1).tolist()
        # calibration[cam].imgSize = [cameras[cam].H, cameras[cam].W]
        calibration[cam].imgSize = args.h_w_overwrite
        calibration[cam].rectifyAlpha = 0.0
    json.dump(calibration, open(join(args.posevocab_root, 'calibration.json'), 'w'))


if __name__ == '__main__':
    main()
