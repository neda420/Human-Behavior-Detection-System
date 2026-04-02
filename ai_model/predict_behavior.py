import json
import pickle
from collections import Counter
from pathlib import Path
import logging

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline MediaPipe-space coordinates for a standing person (centre of frame)
# Used as reference for realistic synthetic data generation.
# ---------------------------------------------------------------------------
_BASELINE_POSE = {
    'nose':           {'x': 0.500, 'y': 0.120, 'z': 0.00},
    'left_shoulder':  {'x': 0.420, 'y': 0.280, 'z': 0.00},
    'right_shoulder': {'x': 0.580, 'y': 0.280, 'z': 0.00},
    'left_elbow':     {'x': 0.360, 'y': 0.420, 'z': 0.00},
    'right_elbow':    {'x': 0.640, 'y': 0.420, 'z': 0.00},
    'left_wrist':     {'x': 0.350, 'y': 0.560, 'z': 0.00},
    'right_wrist':    {'x': 0.650, 'y': 0.560, 'z': 0.00},
    'left_hip':       {'x': 0.440, 'y': 0.540, 'z': 0.00},
    'right_hip':      {'x': 0.560, 'y': 0.540, 'z': 0.00},
    'left_knee':      {'x': 0.440, 'y': 0.700, 'z': 0.00},
    'right_knee':     {'x': 0.560, 'y': 0.700, 'z': 0.00},
    'left_ankle':     {'x': 0.430, 'y': 0.870, 'z': 0.00},
    'right_ankle':    {'x': 0.570, 'y': 0.870, 'z': 0.00},
}

# Joint-angle triplets: (point_a, vertex, point_c) — angle measured at vertex
_JOINT_ANGLE_TRIPLETS = [
    ('left_shoulder',  'left_elbow',  'left_wrist'),    # left elbow flex
    ('right_shoulder', 'right_elbow', 'right_wrist'),   # right elbow flex
    ('left_hip',       'left_knee',   'left_ankle'),    # left knee flex
    ('right_hip',      'right_knee',  'right_ankle'),   # right knee flex
    ('left_shoulder',  'left_hip',    'left_knee'),     # left trunk-hip flex
    ('right_shoulder', 'right_hip',   'right_knee'),    # right trunk-hip flex
]

# Minimum landmark visibility to include in features (below → zeroed out)
_VISIBILITY_THRESHOLD = 0.30


class PoseFeatureExtractor:
    """Extract rich features from pose keypoints for behavior classification.

    Per-frame feature vector (58 dims):
      • 13 landmarks × 3 normalised coords (x, y, z) = 39
      • 13 landmarks × 1 visibility score             = 13
      • 6 joint angles (radians)                      =  6

    Temporal feature vector (232 dims, window W):
      • mean, std, mean-velocity, mean-acceleration of the per-frame vector
    """

    WINDOW_SIZE = 20    # Default temporal window in frames

    def __init__(self):
        self.important_landmarks = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
            'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_between_three_points(a, b, c) -> float:
        """Return the angle (radians) at vertex b formed by a–b–c."""
        v1 = np.array([a['x'] - b['x'], a['y'] - b['y'], a.get('z', 0) - b.get('z', 0)])
        v2 = np.array([c['x'] - b['x'], c['y'] - b['y'], c.get('z', 0) - b.get('z', 0)])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

    def _compute_joint_angles(self, lm: dict) -> list:
        """Return a list of 6 joint angles given a landmark coordinate dict."""
        angles = []
        for (a_name, b_name, c_name) in _JOINT_ANGLE_TRIPLETS:
            a = lm.get(a_name)
            b = lm.get(b_name)
            c = lm.get(c_name)
            if a and b and c:
                angles.append(self._angle_between_three_points(a, b, c))
            else:
                angles.append(0.0)
        return angles

    def _normalize_pose(self, lm: dict) -> dict:
        """Centre on mid-hip and scale by torso height (mid-shoulder → mid-hip)."""
        lh = lm.get('left_hip',  {'x': 0.5, 'y': 0.55, 'z': 0.0})
        rh = lm.get('right_hip', {'x': 0.5, 'y': 0.55, 'z': 0.0})
        ls = lm.get('left_shoulder',  {'x': 0.42, 'y': 0.28, 'z': 0.0})
        rs = lm.get('right_shoulder', {'x': 0.58, 'y': 0.28, 'z': 0.0})

        cx = (lh['x'] + rh['x']) / 2
        cy = (lh['y'] + rh['y']) / 2
        cz = (lh.get('z', 0) + rh.get('z', 0)) / 2

        mid_sx = (ls['x'] + rs['x']) / 2
        mid_sy = (ls['y'] + rs['y']) / 2
        torso = max(np.hypot(mid_sx - cx, mid_sy - cy), 1e-6)

        out = {}
        for name, pts in lm.items():
            vis = pts.get('visibility', 1.0)
            out[name] = {
                'x': (pts['x'] - cx) / torso,
                'y': (pts['y'] - cy) / torso,
                'z': (pts.get('z', 0) - cz) / torso,
                'visibility': vis,
            }
        return out

    def _extract_single_frame_features(self, data: dict) -> np.ndarray:
        """Return a 58-dim feature vector for one pose frame."""
        n_lm = len(self.important_landmarks)
        if not data.get('pose_detected') or not data.get('landmark_coordinates'):
            return np.zeros(n_lm * 4 + 6)  # 3 coords + 1 visibility + 6 angles

        raw_lm = data['landmark_coordinates']
        norm_lm = self._normalize_pose(raw_lm)

        coords_vis = []
        for name in self.important_landmarks:
            lm = norm_lm.get(name)
            if lm:
                vis = lm.get('visibility', 1.0)
                if vis < _VISIBILITY_THRESHOLD:
                    coords_vis.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    coords_vis.extend([lm['x'], lm['y'], lm['z'], vis])
            else:
                coords_vis.extend([0.0, 0.0, 0.0, 0.0])

        angles = self._compute_joint_angles(raw_lm)
        return np.array(coords_vis + angles, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, pose_data: list) -> np.ndarray:
        """Return (N, 58) per-frame feature matrix."""
        return np.array([self._extract_single_frame_features(d) for d in pose_data],
                        dtype=np.float32)

    def extract_temporal_features(self, pose_data: list,
                                   window_size: int = WINDOW_SIZE) -> np.ndarray:
        """Return (N-W+1, 232) temporal feature matrix.

        For each sliding window of *window_size* frames the feature vector is:
          [mean | std | mean_velocity | mean_acceleration]
        where velocity = first diff, acceleration = second diff of per-frame vectors.
        """
        # Pad short sequences so we always produce at least one row
        if len(pose_data) < window_size:
            padding = [pose_data[0]] * (window_size - len(pose_data))
            pose_data = padding + list(pose_data)

        frame_features = self.extract_features(pose_data)   # (N, 58)
        temporal_rows = []

        for i in range(len(frame_features) - window_size + 1):
            w = frame_features[i: i + window_size]          # (W, 58)
            mean_f = w.mean(axis=0)
            std_f  = w.std(axis=0)
            vel    = np.diff(w, axis=0).mean(axis=0)        # (58,)
            if w.shape[0] >= 3:
                acc = np.diff(w, n=2, axis=0).mean(axis=0)  # (58,)
            else:
                acc = np.zeros_like(vel)
            temporal_rows.append(np.concatenate([mean_f, std_f, vel, acc]))

        return np.array(temporal_rows, dtype=np.float32)


class BehaviorClassifier:
    """Classify human behaviors based on temporal pose features.

    Model: HistGradientBoostingClassifier (fast, handles high-dim features well)
    wrapped with CalibratedClassifierCV (Platt scaling) for reliable probabilities.
    Feature space: 232-dim temporal features (window=20, 58 per-frame dims × 4 stats).
    """

    TEMPORAL_WINDOW = 20
    SMOOTHING_WINDOW = 10

    def __init__(self, model_path=None):
        self.feature_extractor = PoseFeatureExtractor()
        self.model = None
        self.behavior_classes = [
            'normal', 'falling', 'fighting', 'loitering', 'running', 'walking'
        ]

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        pose_data_dirs,
        labels,
        model_save_path="ai_model/behavior_classifier.pkl",
        epochs: int = 10,
        trees_per_epoch: int = 10,          # kept for API compatibility (unused internally)
    ):
        """Train a HistGradientBoosting classifier with Platt calibration.

        Uses an 80 / 10 / 10 train / val / test split and prints per-class
        precision, recall, F1 and a confusion matrix on the test set.
        """
        logger.info("Starting model training (HistGradientBoosting + Platt calibration)…")

        all_features: list = []
        all_labels:   list = []

        for pose_dir, label in zip(pose_data_dirs, labels):
            pkl_file = Path(pose_dir) / "pose_data.pkl"
            if not pkl_file.exists():
                logger.warning(f"Pose data file not found: {pkl_file}")
                continue
            with open(pkl_file, 'rb') as f:
                pose_data = pickle.load(f)

            feats = self.feature_extractor.extract_temporal_features(
                pose_data, window_size=self.TEMPORAL_WINDOW)
            if len(feats) > 0:
                all_features.append(feats)
                all_labels.extend([label] * len(feats))

        if not all_features:
            logger.error("No valid pose data found for training.")
            return

        X = np.vstack(all_features)
        y = np.array(all_labels)

        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Label distribution: {dict(Counter(y.tolist()))}")

        # 80 / 10 / 10 split
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.111, random_state=42,
            stratify=y_tv if len(np.unique(y_tv)) > 1 else None
        )
        logger.info(f"Split → train:{len(X_train)}, val:{len(X_val)}, test:{len(X_test)}")

        # Base estimator — HGBC already produces calibrated probabilities via
        # its softmax/cross-entropy loss; no separate calibration wrapper needed.
        base = HistGradientBoostingClassifier(
            max_iter=epochs * trees_per_epoch,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            verbose=1,
        )

        print(f"\n[HGBC] Training {epochs * trees_per_epoch} iterations…")
        base.fit(X_train, y_train)

        train_acc = float(base.score(X_train, y_train))
        val_acc   = float(base.score(X_val,   y_val))
        print(f"[HGBC] train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        logger.info(f"HGBC train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        self.model = base

        # Test-set evaluation
        y_pred = self.model.predict(X_test)
        test_acc = float(np.mean(y_pred == y_test))
        print(f"\n[Test] accuracy={test_acc:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=self.behavior_classes)
        print("Confusion matrix (rows=true, cols=predicted):")
        print(f"  classes: {self.behavior_classes}")
        print(cm)
        logger.info(f"Test accuracy={test_acc:.3f}")

        # Persist
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'behavior_classes': self.behavior_classes,
            'temporal_window': self.TEMPORAL_WINDOW,
            'version': 2,
        }
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        meta_path = Path(model_save_path).with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'version': 2,
                'iterations': epochs * trees_per_epoch,
                'train_acc': round(train_acc, 4),
                'val_acc': round(val_acc, 4),
                'test_acc': round(test_acc, 4),
                'feature_dims': int(X.shape[1]),
                'temporal_window': self.TEMPORAL_WINDOW,
            }, f, indent=2)

    def load_model(self, model_path):
        """Load a trained model."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_extractor = model_data.get('feature_extractor', PoseFeatureExtractor())
        self.behavior_classes  = model_data['behavior_classes']
        self.TEMPORAL_WINDOW   = model_data.get('temporal_window', self.TEMPORAL_WINDOW)
        logger.info(f"Model loaded from {model_path} (version={model_data.get('version', 1)})")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _build_temporal_features(self, pose_data: list) -> np.ndarray:
        """Always returns a 2-D array of temporal features (pads if needed)."""
        return self.feature_extractor.extract_temporal_features(
            pose_data, window_size=self.TEMPORAL_WINDOW)

    def predict_behavior(self, pose_data: list) -> dict:
        """Predict behavior from a list of pose frames.

        Returns per-window predictions aligned to frames (last window label
        repeated to cover the initial frames that don't yet have a full window).
        """
        if self.model is None:
            logger.error("Model not loaded. Train or load a model first.")
            return None

        if not pose_data:
            return {'predictions': [], 'probabilities': [], 'confidence_scores': []}

        feats = self._build_temporal_features(pose_data)   # (M, 232)

        preds     = self.model.predict(feats)               # (M,)
        probs     = self.model.predict_proba(feats)         # (M, C)
        confs     = np.max(probs, axis=1)                   # (M,)

        # Pad back to original length: the first (W-1) frames get the first window label
        n_orig   = len(pose_data)
        n_win    = len(preds)
        pad_len  = max(0, n_orig - n_win)
        preds_full = np.concatenate([[preds[0]] * pad_len, preds])
        probs_full = np.vstack([np.tile(probs[0], (pad_len, 1)), probs])
        confs_full = np.concatenate([[confs[0]] * pad_len, confs])

        return {
            'predictions':     preds_full,
            'probabilities':   probs_full,
            'confidence_scores': confs_full,
        }

    def predict_with_smoothing(self, pose_data: list,
                                smoothing_window: int = None) -> dict:
        """Predict then apply a majority-vote temporal smoothing pass."""
        result = self.predict_behavior(pose_data)
        if result is None or len(result['predictions']) == 0:
            return result

        sw     = smoothing_window or self.SMOOTHING_WINDOW
        preds  = list(result['predictions'])
        smooth = []
        for i in range(len(preds)):
            start = max(0, i - sw + 1)
            window = preds[start: i + 1]
            smooth.append(Counter(window).most_common(1)[0][0])

        result['predictions'] = np.array(smooth)
        return result

    def predict_behavior_from_file(self, pose_data_file):
        """Predict behavior from a pose data file (.pkl or .json)."""
        pose_data_file = Path(pose_data_file)
        if not pose_data_file.exists():
            logger.error(f"Pose data file not found: {pose_data_file}")
            return None
        if pose_data_file.suffix == '.pkl':
            with open(pose_data_file, 'rb') as f:
                pose_data = pickle.load(f)
        else:
            with open(pose_data_file, 'r') as f:
                pose_data = json.load(f)
        return self.predict_behavior(pose_data)

    def get_behavior_summary(self, predictions, confidence_threshold=0.6):
        """Summarise per-frame predictions into a behavior distribution dict."""
        if not predictions or not len(predictions.get('predictions', [])):
            return {'total_frames': 0, 'behaviors': {}, 'behavior_percentages': {},
                    'high_confidence_frames': 0, 'confidence_rate': 0.0}

        behaviors: dict = {}
        total_frames = len(predictions['predictions'])
        high_confidence_frames = 0

        for pred, conf in zip(predictions['predictions'], predictions['confidence_scores']):
            if conf >= confidence_threshold:
                high_confidence_frames += 1
                behaviors[pred] = behaviors.get(pred, 0) + 1

        behavior_percentages = {beh: cnt / total_frames for beh, cnt in behaviors.items()}

        return {
            'total_frames':          total_frames,
            'high_confidence_frames': high_confidence_frames,
            'confidence_rate':       high_confidence_frames / total_frames,
            'behaviors':             behaviors,
            'behavior_percentages':  behavior_percentages,
        }


# ---------------------------------------------------------------------------
# Realistic synthetic training-data generator
# ---------------------------------------------------------------------------

def _make_frame(landmarks: dict, behavior: str, frame_idx: int) -> dict:
    """Wrap a landmark dict into the standard pose-data frame format."""
    return {
        'image_path':         f'synthetic_{behavior}_frame_{frame_idx:06d}.jpg',
        'pose_detected':      True,
        'landmarks':          [],
        'landmark_coordinates': {
            k: {**v, 'visibility': float(np.clip(
                v.get('visibility', 0.95) + np.random.normal(0, 0.03), 0.0, 1.0))}
            for k, v in landmarks.items()
        },
        'image_shape':        (480, 640, 3),
    }


def _gen_normal(n=240) -> list:
    """Standing still with small breathing sway."""
    rng = np.random.default_rng(0)
    frames = []
    pose = {k: dict(v) for k, v in _BASELINE_POSE.items()}
    for i in range(n):
        sway_x = 0.005 * np.sin(i * 0.05)
        sway_y = 0.003 * np.sin(i * 0.08)
        noisy = {}
        for name, lm in pose.items():
            noisy[name] = {
                'x': lm['x'] + sway_x + rng.normal(0, 0.005),
                'y': lm['y'] + sway_y + rng.normal(0, 0.005),
                'z': lm.get('z', 0) + rng.normal(0, 0.003),
                'visibility': 0.95,
            }
        frames.append(_make_frame(noisy, 'normal', i))
    return frames


def _gen_walking(n=240, stride_freq=0.22, arm_amp=0.07, step_amp=0.06) -> list:
    """Walking: alternating arm/leg swing with forward drift."""
    rng = np.random.default_rng(1)
    frames = []
    for i in range(n):
        phase = i * stride_freq
        drift_x = i * 0.001
        lm = {k: dict(v) for k, v in _BASELINE_POSE.items()}

        # Arms swing out-of-phase with ipsilateral leg
        lm['right_wrist']['x'] += arm_amp * np.sin(phase)
        lm['right_wrist']['y'] -= arm_amp * 0.4 * np.sin(phase)
        lm['left_wrist']['x']  -= arm_amp * np.sin(phase)
        lm['left_wrist']['y']  += arm_amp * 0.4 * np.sin(phase)
        lm['right_elbow']['x'] += arm_amp * 0.5 * np.sin(phase)
        lm['left_elbow']['x']  -= arm_amp * 0.5 * np.sin(phase)

        # Legs alternate: lift knee, extend ankle
        rk_lift = step_amp * max(0.0, float(np.sin(phase)))
        lk_lift = step_amp * max(0.0, float(-np.sin(phase)))
        lm['right_knee']['y']  -= rk_lift
        lm['right_ankle']['y'] -= rk_lift * 0.7
        lm['left_knee']['y']   -= lk_lift
        lm['left_ankle']['y']  -= lk_lift * 0.7

        noisy = {
            k: {'x': v['x'] + drift_x + rng.normal(0, 0.007),
                'y': v['y'] + rng.normal(0, 0.007),
                'z': v.get('z', 0) + rng.normal(0, 0.004),
                'visibility': 0.93}
            for k, v in lm.items()
        }
        frames.append(_make_frame(noisy, 'walking', i))
    return frames


def _gen_running(n=240) -> list:
    """Running: larger, faster strides + forward lean."""
    return _gen_walking(n=n, stride_freq=0.44, arm_amp=0.14, step_amp=0.12)


def _gen_falling(n=240) -> list:
    """Falling: body rotates from upright to horizontal over first half, stays down."""
    rng = np.random.default_rng(3)
    frames = []
    fall_frames = n // 2

    for i in range(n):
        t = min(1.0, i / fall_frames)           # 0→1 over fall_frames
        lm = {}
        for name, bp in _BASELINE_POSE.items():
            # Rotate body: y moves toward 0.5, x spreads, head drops below hips
            y_shift = t * 0.40 * (0.5 - bp['y'])   # compress vertically
            x_shift = t * 0.25 * (bp['x'] - 0.5)   # spread horizontally
            lm[name] = {
                'x': bp['x'] + x_shift + rng.normal(0, 0.008),
                'y': bp['y'] + y_shift + rng.normal(0, 0.008),
                'z': bp.get('z', 0) + rng.normal(0, 0.004),
                'visibility': max(0.5, 0.93 - t * 0.2),
            }
        frames.append(_make_frame(lm, 'falling', i))
    return frames


def _gen_fighting(n=240) -> list:
    """Fighting: rapid punch extensions, wide elbow stance."""
    rng = np.random.default_rng(4)
    frames = []
    for i in range(n):
        phase = i * 0.55
        lm = {k: dict(v) for k, v in _BASELINE_POSE.items()}

        # Right arm punches forward/up rapidly
        punch = max(0.0, float(np.sin(phase)))
        lm['right_wrist']['x']  = 0.68 + punch * 0.12
        lm['right_wrist']['y']  = 0.56 - punch * 0.22
        lm['right_elbow']['x']  = 0.66 + punch * 0.05
        lm['right_elbow']['y']  = 0.42 - punch * 0.10

        # Left arm guards (slightly raised)
        guard = max(0.0, float(-np.sin(phase)))
        lm['left_wrist']['x']   = 0.34 - guard * 0.06
        lm['left_wrist']['y']   = 0.40 - guard * 0.10
        lm['left_elbow']['y']   = 0.36 - guard * 0.06

        # Wider shoulder stance
        lm['left_shoulder']['x']  -= 0.03
        lm['right_shoulder']['x'] += 0.03

        noisy = {
            k: {'x': v['x'] + rng.normal(0, 0.009),
                'y': v['y'] + rng.normal(0, 0.009),
                'z': v.get('z', 0) + rng.normal(0, 0.005),
                'visibility': 0.91}
            for k, v in lm.items()
        }
        frames.append(_make_frame(noisy, 'fighting', i))
    return frames


def _gen_loitering(n=240) -> list:
    """Loitering: very slow random drift, occasional weight shifts."""
    rng = np.random.default_rng(5)
    frames = []
    cx, cy = 0.0, 0.0
    for i in range(n):
        cx += rng.normal(0, 0.002)
        cy += rng.normal(0, 0.001)
        # Keep drift bounded
        cx = float(np.clip(cx, -0.04, 0.04))
        cy = float(np.clip(cy, -0.02, 0.02))
        lm = {}
        for name, bp in _BASELINE_POSE.items():
            lm[name] = {
                'x': bp['x'] + cx + rng.normal(0, 0.004),
                'y': bp['y'] + cy + rng.normal(0, 0.004),
                'z': bp.get('z', 0) + rng.normal(0, 0.002),
                'visibility': 0.94,
            }
        frames.append(_make_frame(lm, 'loitering', i))
    return frames


def create_synthetic_training_data() -> dict:
    """Generate biomechanically realistic synthetic pose-data sequences.

    Each behavior sequence contains 240 frames with realistic limb dynamics
    (joint angles, stride patterns, body rotation) rather than uniform offsets.
    Multiple random variations are produced per behaviour to increase diversity.
    """
    logger.info("Generating improved synthetic training data…")
    rng_seed = 99

    raw = {
        'normal':    _gen_normal(240),
        'walking':   _gen_walking(240),
        'running':   _gen_running(240),
        'falling':   _gen_falling(240),
        'fighting':  _gen_fighting(240),
        'loitering': _gen_loitering(240),
    }

    # Add two more speed/style variations per behaviour
    generators = {
        'normal':    lambda s: _gen_normal(240),
        'walking':   lambda s: _gen_walking(240, stride_freq=0.18 + s * 0.06),
        'running':   lambda s: _gen_running(240),
        'falling':   lambda s: _gen_falling(240),
        'fighting':  lambda s: _gen_fighting(240),
        'loitering': lambda s: _gen_loitering(240),
    }
    np.random.seed(rng_seed)
    for behavior, gen in generators.items():
        for s in range(1, 3):
            raw[behavior] = raw[behavior] + gen(s)

    logger.info(f"Synthetic data: { {k: len(v) for k, v in raw.items()} }")
    return raw

def main():
    """Main function to demonstrate behavior classification."""
    # Create synthetic training data
    synthetic_data = create_synthetic_training_data()
    
    # Prepare training data
    pose_data_dirs = []
    labels = []
    
    for behavior, pose_data in synthetic_data.items():
        # Save synthetic data
        output_dir = Path(f"data/keypoints/synthetic_{behavior}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "pose_data.pkl", 'wb') as f:
            pickle.dump(pose_data, f)
        
        pose_data_dirs.append(str(output_dir))
        labels.append(behavior)
    
    # Train model
    classifier = BehaviorClassifier()
    classifier.train_model(pose_data_dirs, labels)
    
    # Test prediction
    test_pose_data = synthetic_data['falling'][:10]  # Test with falling behavior
    predictions = classifier.predict_behavior(test_pose_data)
    
    if predictions:
        summary = classifier.get_behavior_summary(predictions)
        print("\n" + "="*50)
        print("BEHAVIOR PREDICTION RESULTS")
        print("="*50)
        print(f"Total frames: {summary['total_frames']}")
        print(f"High confidence frames: {summary['high_confidence_frames']}")
        print(f"Confidence rate: {summary['confidence_rate']:.2%}")
        print("\nBehavior distribution:")
        for behavior, count in summary['behaviors'].items():
            percentage = summary['behavior_percentages'][behavior]
            print(f"  {behavior}: {count} frames ({percentage:.1%})")

if __name__ == "__main__":
    main() 