import numpy as np
import json
import pickle
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseFeatureExtractor:
    """Extract features from pose keypoints for behavior classification."""
    
    def __init__(self):
        # Define important landmarks for behavior analysis
        self.important_landmarks = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 
            'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def extract_features(self, pose_data):
        """
        Extract features from pose data.
        
        Args:
            pose_data (list): List of pose data dictionaries
            
        Returns:
            np.array: Feature matrix
        """
        features = []
        
        for data in pose_data:
            if not data['pose_detected'] or not data['landmark_coordinates']:
                # If no pose detected, use zero features
                feature_vector = np.zeros(len(self.important_landmarks) * 3)
                features.append(feature_vector)
                continue
            
            feature_vector = []
            landmarks = data['landmark_coordinates']
            
            for landmark_name in self.important_landmarks:
                landmark = landmarks.get(landmark_name)
                if landmark:
                    feature_vector.extend([landmark['x'], landmark['y'], landmark['z']])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_temporal_features(self, pose_data, window_size=5):
        """
        Extract temporal features from pose data.
        
        Args:
            pose_data (list): List of pose data dictionaries
            window_size (int): Size of temporal window
            
        Returns:
            np.array: Temporal feature matrix
        """
        if len(pose_data) < window_size:
            return None
        
        temporal_features = []
        
        for i in range(len(pose_data) - window_size + 1):
            window_data = pose_data[i:i+window_size]
            window_features = self.extract_features(window_data)
            
            # Calculate temporal features
            mean_features = np.mean(window_features, axis=0)
            std_features = np.std(window_features, axis=0)
            velocity_features = np.diff(window_features, axis=0).mean(axis=0)
            
            # Combine features
            combined_features = np.concatenate([mean_features, std_features, velocity_features])
            temporal_features.append(combined_features)
        
        return np.array(temporal_features)

class BehaviorClassifier:
    """Classify human behaviors based on pose features."""
    
    def __init__(self, model_path=None):
        self.feature_extractor = PoseFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.behavior_classes = [
            'normal', 'falling', 'fighting', 'loitering', 'running', 'walking'
        ]
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train_model(
        self,
        pose_data_dirs,
        labels,
        model_save_path="ai_model/behavior_classifier.pkl",
        epochs: int = 10,
        trees_per_epoch: int = 10,
    ):
        """
        Train the behavior classification model.
        
        Args:
            pose_data_dirs (list): List of directories containing pose data
            labels (list): List of behavior labels corresponding to each directory
            model_save_path (str): Path to save the trained model
            epochs (int): Number of training epochs to display/log
            trees_per_epoch (int): Number of trees to add per epoch
        """
        logger.info("Starting model training...")
        
        all_features = []
        all_labels = []
        
        # Load and process pose data
        for pose_dir, label in zip(pose_data_dirs, labels):
            pose_data_file = Path(pose_dir) / "pose_data.pkl"
            
            if not pose_data_file.exists():
                logger.warning(f"Pose data file not found: {pose_data_file}")
                continue
            
            # Load pose data
            with open(pose_data_file, 'rb') as f:
                pose_data = pickle.load(f)
            
            # Extract features
            features = self.feature_extractor.extract_features(pose_data)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.extend([label] * len(features))
        
        if not all_features:
            logger.error("No valid pose data found for training")
            return
        
        # Combine all features
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        logger.info(f"Training data shape: {X.shape}")
        # Compute label distribution safely when labels are strings
        try:
            from collections import Counter
            label_counts = Counter(y.tolist() if hasattr(y, 'tolist') else list(y))
            logger.info(f"Label distribution: {dict(label_counts)}")
        except Exception:
            logger.info("Label distribution: unavailable")
        
        # Scale features
        # If resuming from an existing model, we reuse its scaler to keep feature scaling consistent
        start_trees = 0
        if Path(model_save_path).exists():
            try:
                self.load_model(model_save_path)
                # Ensure warm_start and oob for continued training
                if hasattr(self.model, 'warm_start'):
                    self.model.warm_start = True
                start_trees = getattr(self.model, 'n_estimators', 0) or 0
                logger.info(f"Resuming RF training from {start_trees} trees")
            except Exception:
                logger.warning("Failed to load existing RF model; starting fresh")
        
        X_scaled = self.scaler.fit_transform(X) if self.model is None else self.scaler.transform(X)
        
        # Train model with epoch-like progress using warm_start
        total_trees = start_trees + epochs * trees_per_epoch
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=start_trees,
                warm_start=True,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                oob_score=True,
            )
        
        for epoch in range(1, epochs + 1):
            self.model.n_estimators = getattr(self.model, 'n_estimators', 0) + trees_per_epoch
            self.model.fit(X_scaled, y)
            try:
                train_acc = float(self.model.score(X_scaled, y))
            except Exception:
                train_acc = float('nan')
            oob = getattr(self.model, 'oob_score_', None)
            if oob is not None:
                logger.info(
                    f"Epoch {epoch}/{epochs}: trees={self.model.n_estimators}/{total_trees}, train_acc={train_acc:.3f}, oob={oob:.3f}"
                )
                print(
                    f"[RF] Epoch {epoch}/{epochs} — trees {self.model.n_estimators}/{total_trees} — train_acc {train_acc:.3f} — oob {oob:.3f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}/{epochs}: trees={self.model.n_estimators}/{total_trees}, train_acc={train_acc:.3f}"
                )
                print(
                    f"[RF] Epoch {epoch}/{epochs} — trees {self.model.n_estimators}/{total_trees} — train_acc {train_acc:.3f}"
                )
            # Save checkpoint every epoch
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_extractor': self.feature_extractor,
                'behavior_classes': self.behavior_classes,
            }
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, model_save_path)
            # Save simple meta file to track progress
            try:
                import json
                meta_path = Path(model_save_path).with_suffix('.meta.json')
                meta = {
                    'trees_trained': int(getattr(self.model, 'n_estimators', 0)),
                    'epochs_trained': epoch,
                    'total_trees_target': int(total_trees),
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'behavior_classes': self.behavior_classes
        }
        
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_save_path)
        
        logger.info(f"Model saved to {model_save_path}")
        
        # Print feature importance
        feature_importance = self.model.feature_importances_
        logger.info("Top 10 most important features:")
        for i in np.argsort(feature_importance)[-10:]:
            logger.info(f"  Feature {i}: {feature_importance[i]:.4f}")
    
    def load_model(self, model_path):
        """Load a trained model."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_extractor = model_data['feature_extractor']
        self.behavior_classes = model_data['behavior_classes']
        logger.info(f"Model loaded from {model_path}")
    
    def predict_behavior(self, pose_data):
        """
        Predict behavior from pose data.
        
        Args:
            pose_data (list): List of pose data dictionaries
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            logger.error("Model not loaded. Please train or load a model first.")
            return None
        
        # Extract features
        features = self.feature_extractor.extract_features(pose_data)
        
        if len(features) == 0:
            return {
                'predictions': [],
                'probabilities': [],
                'confidence_scores': []
            }
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Calculate confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores
        }
    
    def predict_behavior_from_file(self, pose_data_file):
        """
        Predict behavior from pose data file.
        
        Args:
            pose_data_file (str): Path to pose data file
            
        Returns:
            dict: Prediction results
        """
        pose_data_file = Path(pose_data_file)
        
        if not pose_data_file.exists():
            logger.error(f"Pose data file not found: {pose_data_file}")
            return None
        
        # Load pose data
        if pose_data_file.suffix == '.pkl':
            with open(pose_data_file, 'rb') as f:
                pose_data = pickle.load(f)
        else:
            with open(pose_data_file, 'r') as f:
                pose_data = json.load(f)
        
        return self.predict_behavior(pose_data)
    
    def get_behavior_summary(self, predictions, confidence_threshold=0.7):
        """
        Get a summary of predicted behaviors.
        
        Args:
            predictions (dict): Prediction results
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            dict: Behavior summary
        """
        if not predictions or not predictions['predictions']:
            return {'total_frames': 0, 'behaviors': {}}
        
        behaviors = {}
        total_frames = len(predictions['predictions'])
        high_confidence_frames = 0
        
        for i, (pred, conf) in enumerate(zip(predictions['predictions'], predictions['confidence_scores'])):
            if conf >= confidence_threshold:
                high_confidence_frames += 1
                behaviors[pred] = behaviors.get(pred, 0) + 1
        
        # Calculate percentages
        behavior_percentages = {}
        for behavior, count in behaviors.items():
            behavior_percentages[behavior] = count / total_frames
        
        return {
            'total_frames': total_frames,
            'high_confidence_frames': high_confidence_frames,
            'confidence_rate': high_confidence_frames / total_frames,
            'behaviors': behaviors,
            'behavior_percentages': behavior_percentages
        }

def create_synthetic_training_data():
    """Create synthetic training data for demonstration purposes."""
    logger.info("Creating synthetic training data...")
    
    # Create synthetic pose data for different behaviors
    behaviors = ['normal', 'falling', 'fighting', 'loitering', 'running', 'walking']
    synthetic_data = {}
    
    for behavior in behaviors:
        # Generate synthetic pose data
        num_frames = 50
        pose_data = []
        
        for i in range(num_frames):
            # Create synthetic landmarks
            landmarks = {}
            for landmark_name in PoseFeatureExtractor().important_landmarks:
                # Add some randomness based on behavior
                if behavior == 'falling':
                    # Simulate falling motion
                    y_offset = 0.1 * (i / num_frames)  # Gradually move down
                elif behavior == 'fighting':
                    # Simulate fighting motion
                    x_offset = 0.05 * np.sin(i * 0.5)  # Side-to-side motion
                elif behavior == 'running':
                    # Simulate running motion
                    x_offset = 0.03 * i  # Forward motion
                else:
                    x_offset = 0
                    y_offset = 0
                
                landmarks[landmark_name] = {
                    'x': 0.5 + x_offset + np.random.normal(0, 0.02),
                    'y': 0.5 + y_offset + np.random.normal(0, 0.02),
                    'z': 0.0 + np.random.normal(0, 0.01),
                    'visibility': 0.9 + np.random.normal(0, 0.1)
                }
            
            pose_data.append({
                'image_path': f'synthetic_{behavior}_frame_{i:06d}.jpg',
                'pose_detected': True,
                'landmarks': [],
                'landmark_coordinates': landmarks,
                'image_shape': (480, 640, 3)
            })
        
        synthetic_data[behavior] = pose_data
    
    return synthetic_data

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