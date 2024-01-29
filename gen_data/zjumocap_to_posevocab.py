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
from easyvolcap.utils.data_utils import load_dotdict, export_dotdict, to_tensor, export_pts, export_mesh
from easyvolcap.utils.easy_utils import read_camera, write_camera

from easymocap.bodymodel.smpl import SMPLModel
from smplx.body_models import SMPL, SMPLX
import torch


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
    # easymocap: R @ lbs + T
    # smplx: lbs + T
    model = SMPLModel('/home/xuzhen/code/easymocap-public/data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl', NUM_SHAPES=10)
    vertices = model(poses=motion.poses[..., :72], shapes=motion.shapes[..., :10], Rh=motion.Rh, Th=motion.Th)
    motion.poses = motion.poses[:, 3:72]
    motion.shapes = motion.shapes[:, :10]
    motion = dotdict(model.convert_to_standard_smpl(motion))

    smpl = SMPL('/home/xuzhen/code/easymocap-public/data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')
    tensor = to_tensor(motion)
    spl = smpl(tensor.shapes[:, :10], tensor.poses[:, 3:72], tensor.poses[:, :3], tensor.Th)

    smplx = SMPLX('/home/xuzhen/code/posevocab/smpl_files/smplx', use_pca=False)
    spx = smplx(betas=tensor.shapes[:, :10],
                body_pose=tensor.poses[:, 3:72 - 6],
                global_orient=tensor.poses[:, :3],
                transl=tensor.Th,
                expression=smplx.expression.repeat(tensor.shapes.shape[0], 1),
                jaw_pose=smplx.jaw_pose.repeat(tensor.shapes.shape[0], 1),
                leye_pose=smplx.leye_pose.repeat(tensor.shapes.shape[0], 1),
                reye_pose=smplx.reye_pose.repeat(tensor.shapes.shape[0], 1),
                left_hand_pose=smplx.left_hand_pose.repeat(tensor.shapes.shape[0], 1),
                right_hand_pose=smplx.right_hand_pose.repeat(tensor.shapes.shape[0], 1),
                )

    export_pts(vertices[0], filename='emc.ply')
    export_pts(spl.vertices[0], filename='spl.ply')
    export_pts(spx.vertices[0], filename='spx.ply')

    smpl_params = dotdict()
    smpl_params.global_orient = motion.poses[:, :3]
    smpl_params.transl = motion.Th
    smpl_params.body_pose = motion.poses[:, 3:72 - 6]  # 21 * 3
    smpl_params.betas = np.zeros_like(motion.shapes[:, :10])
    # smpl_params = dotdict()
    # smpl_params.global_orient = motion.Rh
    # smpl_params.transl = motion.Th
    # smpl_params.body_pose = motion.poses[:, 3:72 - 6]  # 21 * 3
    # smpl_params.betas = motion.shapes[:, :10]
    export_dotdict(smpl_params, join(args.posevocab_root, args.smpl_params_file))

    cameras = read_camera(join(args.zjumocap_root))
    calibration = dotdict()
    for cam in cameras:
        calibration[cam] = dotdict()
        calibration[cam].K = cameras[cam].K.reshape(-1).tolist()
        calibration[cam].R = cameras[cam].R.reshape(-1).tolist()
        calibration[cam].T = cameras[cam].T.reshape(-1).tolist()
        calibration[cam].distCoeff = cameras[cam].D.reshape(-1).tolist()
        # calibration[cam].imgSize = [cameras[cam].H, cameras[cam].W]
        calibration[cam].imgSize = args.h_w_overwrite
        calibration[cam].rectifyAlpha = 0.0
    json.dump(calibration, open(join(args.posevocab_root, 'calibration.json'), 'w'))


if __name__ == '__main__':
    main()
