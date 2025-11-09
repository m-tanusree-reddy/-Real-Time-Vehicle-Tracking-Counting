"""
This module provides an implementation of the SORT (Simple Online and Realtime Tracking)
algorithm, a pragmatic approach for multi-object tracking with a focus on simplicity and speed.

The algorithm uses a Kalman filter for motion prediction and the Hungarian algorithm for data
association. It's designed to be effective for tracking objects like pedestrians or vehicles in
video streams.

Core Components:
- KalmanBoxTracker: Manages the state of a single tracked object using a Kalman filter.
- Sort: The main class that orchestrates the tracking process across multiple objects and frames.
- Helper functions: For calculating Intersection over Union (IoU) and associating detections
  to trackers.
"""
from typing import List, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) for a single pair of bounding boxes.

    Args:
        bb_test (np.ndarray): Bounding box [x1, y1, x2, y2].
        bb_gt (np.ndarray): Ground truth bounding box [x1, y1, x2, y2].

    Returns:
        float: The IoU value.
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = max(0.0, (bb_test[2] - bb_test[0])) * max(0.0, (bb_test[3] - bb_test[1]))
    area2 = max(0.0, (bb_gt[2] - bb_gt[0])) * max(0.0, (bb_gt[3] - bb_gt[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def iou_batch(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
    """
    Computes Intersection over Union (IoU) for two sets of bounding boxes in a vectorized manner.

    Args:
        dets (np.ndarray): Detections, shape (N, 4) where N is the number of detections.
        trks (np.ndarray): Tracked objects, shape (M, 4) where M is the number of trackers.

    Returns:
        np.ndarray: An (N, M) matrix with the IoU values.
    """
    if dets.size == 0 or trks.size == 0:
        return np.zeros((dets.shape[0], trks.shape[0]), dtype=np.float32)

    # Vectorized computation of IoU
    dets_exp = dets[:, None, :]
    trks_exp = trks[None, :, :]

    xx1 = np.maximum(dets_exp[..., 0], trks_exp[..., 0])
    yy1 = np.maximum(dets_exp[..., 1], trks_exp[..., 1])
    xx2 = np.minimum(dets_exp[..., 2], trks_exp[..., 2])
    yy2 = np.minimum(dets_exp[..., 3], trks_exp[..., 3])

    w = np.clip(xx2 - xx1, 0.0, None)
    h = np.clip(yy2 - yy1, 0.0, None)
    inter = w * h

    dets_area = np.clip(dets_exp[..., 2] - dets_exp[..., 0], 0.0, None) * np.clip(
        dets_exp[..., 3] - dets_exp[..., 1], 0.0, None
    )
    trks_area = np.clip(trks_exp[..., 2] - trks_exp[..., 0], 0.0, None) * np.clip(
        trks_exp[..., 3] - trks_exp[..., 1], 0.0, None
    )

    union = dets_area + trks_area - inter
    eps = np.finfo(np.float32).eps
    return np.where(union > 0.0, inter / (union + eps), 0.0).astype(np.float32)


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
    [x, y, s, r] where x,y is the centre of the box, s is the scale/area, and r is
    the aspect ratio.

    Args:
        bbox (np.ndarray): Bounding box in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Kalman filter measurement vector [x, y, s, r].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / (h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in the centre form [x, y, s, r] and returns it in the form
    [x1, y1, x2, y2] where x1, y1 is the top-left and x2, y2 is the bottom-right.

    Args:
        x (np.ndarray): Kalman filter state vector [x, y, s, r, ...].

    Returns:
        np.ndarray: Bounding box in [x1, y1, x2, y2] format.
    """
    x_c, y_c, s, r = x[0], x[1], max(1.0, x[2]), max(1e-6, x[3])
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    It uses a Kalman filter to predict the object's position in subsequent frames.
    """

    _count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initializes a tracker using initial bounding box.

        The state is [x, y, s, r, dx, dy, ds, dr], where (x, y) is the center,
        s is the scale/area, r is the aspect ratio, and the rest are velocities.
        However, the aspect ratio velocity 'dr' is not used in this model.
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        # State Transition Matrix
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # Measurement Matrix
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        # Measurement Noise Covariance
        self.kf.R[2:, 2:] *= 10.0
        # Process Covariance
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        # Process Noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox: np.ndarray):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def get_state(self) -> np.ndarray:
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns detections to tracked object (both represented as bounding boxes).

    Args:
        detections (np.ndarray): A set of detected bounding boxes.
        trackers (np.ndarray): A set of bounding boxes predicted by trackers.
        iou_threshold (float): The IoU threshold for considering a match.

    Returns:
        A tuple of three arrays:
        - matches (np.ndarray): Array of shape (N, 2) with indices of matched (detection, tracker).
        - unmatched_dets (np.ndarray): Array of indices of unmatched detections.
        - unmatched_trks (np.ndarray): Array of indices of unmatched trackers.
    """
    if trackers.size == 0 or detections.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(detections.shape[0]),
            np.arange(trackers.shape[0]),
        )

    ious = iou_batch(detections, trackers)
    cost = 1.0 - ious

    # Use the Hungarian algorithm (linear_sum_assignment) to find optimal assignments
    det_idx, trk_idx = linear_sum_assignment(cost)

    matches = []
    unmatched_dets = set(range(detections.shape[0]))
    unmatched_trks = set(range(trackers.shape[0]))

    for d, t in zip(det_idx, trk_idx):
        if ious[d, t] >= iou_threshold:
            matches.append([d, t])
            unmatched_dets.discard(d)
            unmatched_trks.discard(t)

    return (
        np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int),
        np.array(sorted(list(unmatched_dets)), dtype=int),
        np.array(sorted(list(unmatched_trks)), dtype=int),
    )


class Sort:
    """
    Sort is the main tracking class. It takes object detections for each frame
    and manages the lifecycle of trackers.
    """

    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initializes the Sort tracker.

        Args:
            max_age (int): Maximum number of frames to keep a track alive without new detections.
            min_hits (int): Minimum number of consecutive detections to start a track.
            iou_threshold (float): IoU threshold for matching detections to existing tracks.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0  # Reset tracker ID count

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        This method is the main update loop. It should be called for each frame.

        Args:
            dets (np.ndarray): A numpy array of detections in the format [[x1,y1,x2,y2,score],...].
                               If no detections are present, it should be an empty array.

        Returns:
            np.ndarray: A numpy array of tracked objects in the format [[x1,y1,x2,y2,track_id],...].
        """
        self.frame_count += 1

        # 1. Predict new locations of existing trackers
        trks = []
        for t in self.trackers:
            pos = t.predict()
            trks.append(pos.reshape(-1))
        trks = np.array(trks) if len(trks) > 0 else np.empty((0, 4))

        # 2. Associate detections with predicted tracker locations
        detection_bboxes = dets[:, :4] if dets.size else np.empty((0, 4))
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detection_bboxes, trks, self.iou_threshold
        )

        # 3. Update matched trackers with new detection info
        for m in matches:
            trk_idx = m[1]
            det_idx = m[0]
            self.trackers[trk_idx].update(dets[det_idx, :4])

        # 4. Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))
        
        # 5. Manage tracker lifecycle and prepare output
        ret = []
        alive_trackers = []
        for t in self.trackers:
            # A track is considered confirmed if it has been updated recently and has a sufficient hit streak.
            # We also include tracks early on (before min_hits frames have passed) to get initial results.
            if (t.time_since_update < 1) and (
                t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                d = t.get_state().reshape(-1)
                ret.append(np.concatenate([d, [t.id]], axis=0))
            
            # Remove trackers that have been lost for too long
            if t.time_since_update <= self.max_age:
                alive_trackers.append(t)

        self.trackers = alive_trackers

        return np.array(ret) if len(ret) > 0 else np.empty((0, 5))


if __name__ == "__main__":
    # --- Example Usage ---

    # Create an instance of the Sort tracker
    tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

    # Simulate frames from a video
    # Each frame contains a list of detections: [x1, y1, x2, y2, confidence_score]
    simulated_frames = [
        # Frame 1: One object detected
        np.array([[100, 100, 150, 150, 0.9]]),
        # Frame 2: The object moved slightly
        np.array([[105, 105, 155, 155, 0.92]]),
        # Frame 3: Object continues to move, and a new object appears
        np.array([[110, 110, 160, 160, 0.93], [300, 200, 350, 250, 0.88]]),
        # Frame 4: Both objects move
        np.array([[115, 115, 165, 165, 0.94], [305, 205, 355, 255, 0.89]]),
        # Frame 5: First object is missed by the detector
        np.array([[310, 210, 360, 260, 0.90]]),
        # Frame 6: First object is detected again, second one continues
        np.array([[125, 125, 175, 175, 0.91], [315, 215, 365, 265, 0.91]]),
    ]
    
    print("--- Running SORT Tracker Simulation ---")
    for frame_num, detections in enumerate(simulated_frames):
        print(f"\n--- Frame {frame_num + 1} ---")
        print(f"Detections:\n{detections}")
        
        # Update the tracker with the new detections
        tracked_objects = tracker.update(detections)
        
        print(f"Tracked Objects (Output format: [x1, y1, x2, y2, track_id]):\n{tracked_objects}")
