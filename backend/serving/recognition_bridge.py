"""Bridge into the AI core (src/): adds src/ to sys.path, loads SPOTER transforms."""

import sys

from serving.config import settings

_bootstrapped = False


def _bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    src = str(settings.recognition_src)
    if src not in sys.path:
        sys.path.insert(0, src)
    _bootstrapped = True


def load_transforms():
    """Keypoint normalization only (client already ran Holistic + joint selection)."""
    _bootstrap()
    from torchvision.transforms.v2 import Compose

    from pipelines.spoter_graph_classification import (
        DictToTensor,
        Pad,
        Shift,
        SingleBodyDictNormalize,
        SPOTERSingleHandDictNormalize,
        TensorToDict,
    )

    return Compose(
        [
            TensorToDict(),
            SingleBodyDictNormalize(),
            SPOTERSingleHandDictNormalize(),
            DictToTensor(),
            Shift(),
            Pad(settings.spoter_seq_len),
        ]
    )


def load_pose_extract():
    """PoseExtract (MediaPipe Holistic): dict -> Pose. Video-file flow only (one
    extraction per segment, shared by SPOTER and SL-GCN); realtime doesn't use this
    since the client already runs Holistic and sends keypoints."""
    _bootstrap()
    from pipelines.spoter_graph_classification import PoseExtract

    return PoseExtract()


def load_spoter_graph_transforms():
    """SPOTER branch for the video flow: Pose -> tensor (spoter_seq_len, 54, 2)."""
    _bootstrap()
    from torchvision.transforms.v2 import Compose

    from pipelines.spoter_graph_classification import (
        DictToTensor,
        JointSelect,
        Pad,
        Shift,
        SingleBodyDictNormalize,
        SPOTERSingleHandDictNormalize,
        TensorToDict,
    )

    return Compose(
        [
            JointSelect(),
            TensorToDict(),
            SingleBodyDictNormalize(),
            SPOTERSingleHandDictNormalize(),
            DictToTensor(),
            Shift(),
            Pad(settings.spoter_seq_len),
        ]
    )


def load_slgcn_transforms():
    """SL-GCN branch for the video flow: Pose -> array (3, T, num_points, 1). Copies
    SLGCNJointSelect/SLGCNPad from src/features/transforms/sl_gcn.py, omitting
    _build_edges since bone/motion/normalize are already baked into the ONNX graph."""
    _bootstrap()

    import numpy as np
    from pose_format import Pose
    from torchvision.transforms.v2 import Compose

    from utils import COCO_TO_POSE_FORMAT, SLGCN_JOINTS

    class SLGCNJointSelect:
        """Selects num_points joints for SL-GCN; each joint is (x, y, confidence)."""

        def __init__(self, num_points: int) -> None:
            self.joints = SLGCN_JOINTS[num_points]

        def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
            idx = pose.header._get_point_index(component, point)
            T, _, _, C = pose.body.data.shape
            data = np.zeros((T, C), dtype=pose.body.data.dtype)
            data[:, :2] = pose.body.data[:, 0, idx, :2].data
            data[:, 2] = pose.body.confidence[:, 0, idx]
            return data

        def __call__(self, pose: Pose) -> np.ndarray:
            pose.normalize_distribution()
            data = []
            for joint in self.joints:
                component, point = COCO_TO_POSE_FORMAT[joint]
                data.append(self.__get_point(component, point, pose))
            # (num_landmarks, num_frames, 3) -> (num_frames, num_landmarks, 3)
            return np.array(data).transpose((1, 0, 2))

    class SLGCNPad:
        """Pads/crops to num_frames, then transposes to (C, T, V, M) for ONNX SL-GCN."""

        def __init__(self, num_frames: int) -> None:
            self.num_frames = num_frames

        def __call__(self, data: np.ndarray) -> np.ndarray:
            padded_data = np.zeros(
                (self.num_frames, data.shape[1], data.shape[2], 1),
                dtype=np.float32,
            )
            L = data.shape[0]
            if L < self.num_frames:
                padded_data[:L, :, :, 0] = data
                rest = self.num_frames - L
                num = int(np.ceil(rest / L))
                pad = np.concatenate([data for _ in range(num)], 0)[:rest]
                padded_data[L:, :, :, 0] = pad
            else:
                padded_data[:, :, :, 0] = data[: self.num_frames, :, :]
            # (num_frames, num_points, num_channels, num_people)
            # -> (num_channels, num_frames, num_points, num_people)
            return np.transpose(padded_data, [2, 0, 1, 3])

    return Compose(
        [
            SLGCNJointSelect(settings.slgcn_num_points),
            SLGCNPad(settings.slgcn_seq_len),
        ]
    )
