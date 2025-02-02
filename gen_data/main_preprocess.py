import os
import torch
import numpy as np
import pytorch3d.transforms

from easyvolcap.utils.console_utils import *
from gen_data.volume import calc_cano_weight_volume

device = 'cuda:0'


def farthest_point_sampling(points):
    """
    :param points: torch.Tensor, (N, D), D is the dimension of each point
    :return: (N, )
    """
    rest_indices = list(range(points.shape[0]))
    sampled_indices = [0]
    rest_indices.remove(sampled_indices[0])
    while len(rest_indices) > 0:
        rest_points = points[rest_indices]
        sampled_points = points[sampled_indices]
        dot_dist = torch.abs(torch.einsum('vi,mi->vm', rest_points, sampled_points))  # larger is closer
        neg_dot_dist = 1. - dot_dist  # smaller is closer
        min_dist = neg_dot_dist.min(1)[0]
        argmax_pos = min_dist.argmax().item()
        max_idx = rest_indices[argmax_pos]
        sampled_indices.append(max_idx)
        del rest_indices[argmax_pos]
    return sampled_indices


def sample(args):
    smpl_params = np.load(args.data_dir + '/smpl_params.npz')
    body_poses = smpl_params['body_pose'][args.frame_list]
    if body_poses.shape[1] == 69 or body_poses.shape[1] == 63:
        print('# Using smpl data')
        body_poses = body_poses[:, :21 * 3]
    else:
        print('# Using smpl-x data')
    body_poses = torch.from_numpy(body_poses).to(torch.float32).to(device)
    body_poses = body_poses.reshape(-1, 21, 3).permute(1, 0, 2)  # (J, N, 3)
    quaternions = pytorch3d.transforms.axis_angle_to_quaternion(body_poses)  # (J, N, 4)

    sort_indices = []
    sort_quats = []
    for joint_idx in range(quaternions.shape[0]):
        print('Sorting joint %d' % joint_idx)
        sort_indices_ = farthest_point_sampling(quaternions[joint_idx])
        sort_indices.append(sort_indices_)
        sort_quats.append(quaternions[joint_idx][sort_indices_])
    sort_indices = np.array(sort_indices, np.int32)
    sort_quats = torch.stack(sort_quats, 0).cpu().numpy().astype(np.float32)
    os.makedirs(args.data_dir + '/key_rotations', exist_ok=True)
    np.save(args.data_dir + '/key_rotations/sorted_indices.npy', sort_indices)
    np.save(args.data_dir + '/key_rotations/sorted_rotations.npy', sort_quats)


@catch_throw
def main():
    # data_dir, frame_list = 'data/thuman4/subject00', list(range(0, 2000))
    args = dotdict()
    args.data_dir = 'data/my_zju_mocap/my_313'
    args.frame_list = list(range(0, 60))
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    """ sample key rotations """
    sample(args)

    """ calculate blending weight volume """
    calc_cano_weight_volume(args.data_dir, gender='neutral')


if __name__ == '__main__':
    main()
