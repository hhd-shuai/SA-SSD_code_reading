import numpy as np
from mmdet.ops.points_op import points_to_voxel

class VoxelGenerator:                           # config info: val["generator"]
    def __init__(self,
                 voxel_size,                    # [0.05, 0.05, 0.1]
                 point_cloud_range,             # [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
                 max_num_points,                # 5
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32) #[  0.  -40.   -3.   70.4  40.    1. ]
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32) #[0.05 0.05 0.1 ]
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size # [1408. 1600.   40.]
        grid_size = np.round(grid_size).astype(np.int64)
        self._voxel_size = voxel_size #[1408 1600   40]
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size             # array([1408, 1600,   40])

    def generate(self, points):
        return points_to_voxel(
            points, self._voxel_size, self._point_cloud_range,
            self._max_num_points, True, self._max_voxels)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


