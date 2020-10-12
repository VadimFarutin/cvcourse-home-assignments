#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple
from operator import is_

import numpy as np
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    TriangulationParameters,
    build_correspondences,
    create_cli,
    calc_point_cloud_colors,
    eye3x4,
    pose_to_view_mat3x4,
    triangulate_correspondences,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    remove_correspondences_with_ids,
    rodrigues_and_translation_to_view_mat3x4
)


def _track_camera_int(corner_storage,
                      intrinsic_mat,
                      parameters,
                      known_view_1,
                      known_view_2):

    storage_size = len(corner_storage)
    view_mats = np.array([None] * storage_size)

    known_view_idx_1, known_view_idx_2 = known_view_1[0], known_view_2[0]
    view_mats[known_view_idx_1] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_idx_2] = pose_to_view_mat3x4(known_view_2[1])

    points = np.array([None] * (corner_storage.max_corner_id() + 1))

    _add_points(points,
                corner_storage[known_view_idx_1],
                corner_storage[known_view_idx_2],
                view_mats[known_view_idx_1],
                view_mats[known_view_idx_2],
                intrinsic_mat,
                parameters)

    while True:
        is_changed = False

        for i in np.where(np.vectorize(is_)(view_mats, None))[0]:
            print(f"Frame index: {i}")

            mat = _get_pose_mat(points,
                                corner_storage[i.astype(int)],
                                intrinsic_mat)

            if mat is None:
                continue

            view_mats[i] = mat
            is_changed = True

            for j in range(storage_size):
                if view_mats[j] is not None and j != i:
                    _add_points(points,
                                corner_storage[i],
                                corner_storage[j],
                                view_mats[i],
                                view_mats[j],
                                intrinsic_mat,
                                parameters)

        if not is_changed:
            break

    for i in range(known_view_idx_1, known_view_idx_1 + storage_size):
        i = i % storage_size

        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]

    mask = np.vectorize(is_)(points, None) != True
    points = np.array([p for p in points])
    points = points[mask]
    points = np.array([p for p in points])

    point_cloud_builder = PointCloudBuilder(ids=np.where(mask)[0].astype(int),
                                            points=points)

    return view_mats, point_cloud_builder


def _add_points(point_all, corners1, corners2, mat1, mat2, intrinsic_mat, parameters):
    correspondences = build_correspondences(corners1, corners2)
    points, ids, _ = triangulate_correspondences(correspondences,
                                                 mat1,
                                                 mat2,
                                                 intrinsic_mat,
                                                 parameters)

    print(f"Triangulated correspondences: {len(points)}")

    if len(ids) == 0:
        return

    mask = np.array(np.vectorize(is_)(point_all[ids.astype(int)], None))

    ids = ids[mask]
    ids = np.array([p for p in ids])

    points = points[mask]
    points2 = np.array([None] * (len(points)))
    points2[:] = [np.array(p) for p in points]

    point_all[ids.astype(int)] = points2


def _get_pose_mat(point_all, corners, intrinsic_mat):
    ids = corners.ids.flatten()
    mask = np.array(np.vectorize(is_)(point_all[ids.astype(int)], None) != True)
    points = corners.points[mask]

    if len(points) < 5:
        return None

    object_points = np.array([p for p in point_all[ids[mask]]])
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points,
                                                     points,
                                                     intrinsic_mat,
                                                     None)

    print(f"Inliers: {len(inliers)}")

    if not retval:
        return None

    point_all[ids[ids not in inliers.flatten()]] = None
    mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    return mat


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    parameters = TriangulationParameters(max_reprojection_error=1.0,
                                         min_triangulation_angle_deg=2.0,
                                         min_depth=0.1)
    view_mats, point_cloud_builder = _track_camera_int(corner_storage,
                                                       intrinsic_mat,
                                                       parameters,
                                                       known_view_1,
                                                       known_view_2)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
