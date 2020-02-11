#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    min_distance = 10
    corners_cnt = 500
    quality_level = 0.01
    params_dict = dict(maxCorners=corners_cnt,
                       qualityLevel=quality_level,
                       minDistance=min_distance,
                       useHarrisDetector=False,
                       blockSize=min_distance)

    prev_frame = frame_sequence[0]
    prev_frame *= 255
    prev_frame = prev_frame.astype(np.uint8)

    corners = cv2.goodFeaturesToTrack(image=prev_frame, **params_dict)
    ids = np.arange(len(corners))
    sizes = np.full(len(corners), min_distance)

    frame_corners = FrameCorners(ids, corners, sizes)
    builder.set_corners_at_frame(0, frame_corners)

    for frame, cur_frame in enumerate(frame_sequence[1:], 1):
        cur_frame *= 255
        cur_frame = cur_frame.astype(np.uint8)

        corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame,
                                                      corners, None,
                                                      winSize=(min_distance, min_distance))
        status = status.reshape(-1).astype(np.bool)
        corners = corners[status]
        ids = ids[status]
        sizes = np.full(len(corners), min_distance)

        if len(corners) < corners_cnt:
            mask = np.full_like(cur_frame, 255)

            for x, y in corners.reshape(-1, 2):
                cv2.circle(mask, (x, y), min_distance, 0, -1)

            params_dict['maxCorners'] = corners_cnt - len(corners)
            new_corners = cv2.goodFeaturesToTrack(cur_frame, mask=mask, **params_dict)
            corners = np.append(corners, new_corners).reshape((-1, 1, 2))
            ids = np.arange(len(corners))
            sizes = np.full(len(corners), min_distance)

        frame_corners = FrameCorners(ids, corners, sizes)
        builder.set_corners_at_frame(frame, frame_corners)
        prev_frame = cur_frame


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
