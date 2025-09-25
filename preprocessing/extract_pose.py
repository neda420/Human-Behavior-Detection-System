import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseExtractor:
    def __init__(self, input_dir="data/extracted_frames", output_dir="data/keypoints"):
        """
        Initialize the pose extractor using MediaPipe.
        
        Args:
            input_dir (str): Directory containing extracted frames
            output_dir (str): Directory to save pose keypoints
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Define pose landmarks
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
            'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
            'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
            'right_index', 'left_thumb', 'right_thumb', 'left_hip',
            'right_hip', 'left_knee', 'right_knee', 'left_ankle',
            'right_ankle', 'left_heel', 'right_heel', 'left_foot_index',
            'right_foot_index'
        ]
    
    def extract_pose_from_image(self, image_path):
        """
        Extract pose keypoints from a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Pose keypoints data
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
            
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        
        return self.extract_pose_from_ndarray(image, source=str(image_path))
    
    def extract_pose_from_ndarray(self, image: np.ndarray, source: str = "camera_frame"):
        """
        Extract pose keypoints from an already loaded image (numpy array), e.g., webcam frame.
        
        Args:
            image (np.ndarray): BGR image array
            source (str): Optional source identifier
        Returns:
            dict: Pose keypoints data
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid image array provided to extract_pose_from_ndarray")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return {
                'image_path': source,
                'pose_detected': False,
                'landmarks': None,
                'landmark_coordinates': None,
                'image_shape': image.shape
            }
        
        # Extract landmark coordinates
        landmarks = []
        landmark_coordinates = {}
        
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
            landmark_coordinates[self.landmark_names[i]] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return {
            'image_path': source,
            'pose_detected': True,
            'landmarks': landmarks,
            'landmark_coordinates': landmark_coordinates,
            'image_shape': image.shape
        }

    def extract_pose_from_directory(self, frame_dir):
        """
        Extract pose keypoints from all frames in a directory.
        
        Args:
            frame_dir (str): Directory containing frames
            
        Returns:
            dict: Pose extraction results
        """
        frame_dir = Path(frame_dir)
        
        if not frame_dir.exists():
            logger.error(f"Frame directory not found: {frame_dir}")
            return None
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(frame_dir.glob(f"*{ext}"))
        
        if not image_files:
            logger.warning(f"No image files found in {frame_dir}")
            return None
        
        logger.info(f"Processing {len(image_files)} images from {frame_dir.name}")
        
        # Create output subdirectory
        output_subdir = self.output_dir / frame_dir.name
        output_subdir.mkdir(exist_ok=True)
        
        pose_data = []
        successful_extractions = 0
        
        # Process each image
        for image_file in tqdm(image_files, desc=f"Extracting poses from {frame_dir.name}"):
            pose_result = self.extract_pose_from_image(image_file)
            
            if pose_result:
                pose_data.append(pose_result)
                if pose_result['pose_detected']:
                    successful_extractions += 1
        
        # Save results
        output_file = output_subdir / "pose_data.json"
        with open(output_file, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        # Save as pickle for faster loading
        pickle_file = output_subdir / "pose_data.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(pose_data, f)
        
        logger.info(f"Extracted poses from {successful_extractions}/{len(image_files)} images")
        logger.info(f"Results saved to {output_file} and {pickle_file}")
        
        return {
            'frame_dir': str(frame_dir),
            'output_dir': str(output_subdir),
            'total_images': len(image_files),
            'successful_extractions': successful_extractions,
            'pose_data_file': str(output_file),
            'pose_data_pickle': str(pickle_file)
        }
    
    def extract_pose_from_all_directories(self):
        """
        Extract pose keypoints from all frame directories.
        
        Returns:
            list: List of extraction results for each directory
        """
        frame_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not frame_dirs:
            logger.warning(f"No frame directories found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(frame_dirs)} frame directories")
        
        results = []
        for frame_dir in frame_dirs:
            result = self.extract_pose_from_directory(frame_dir)
            if result:
                results.append(result)
        
        return results
    
    def load_pose_data(self, pose_data_file):
        """
        Load pose data from file.
        
        Args:
            pose_data_file (str): Path to pose data file
            
        Returns:
            list: Loaded pose data
        """
        pose_data_file = Path(pose_data_file)
        
        if not pose_data_file.exists():
            logger.error(f"Pose data file not found: {pose_data_file}")
            return None
        
        # Try pickle first, then JSON
        pickle_file = pose_data_file.with_suffix('.pkl')
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        else:
            with open(pose_data_file, 'r') as f:
                return json.load(f)
    
    def get_pose_statistics(self, pose_data):
        """
        Get statistics about pose data.
        
        Args:
            pose_data (list): List of pose data dictionaries
            
        Returns:
            dict: Statistics about the pose data
        """
        if not pose_data:
            return None
        
        total_frames = len(pose_data)
        frames_with_pose = sum(1 for data in pose_data if data['pose_detected'])
        
        # Calculate average visibility for each landmark
        landmark_visibilities = {}
        for landmark_name in self.landmark_names:
            visibilities = []
            for data in pose_data:
                if data['pose_detected'] and data['landmark_coordinates']:
                    landmark = data['landmark_coordinates'].get(landmark_name)
                    if landmark:
                        visibilities.append(landmark['visibility'])
            
            if visibilities:
                landmark_visibilities[landmark_name] = {
                    'mean_visibility': np.mean(visibilities),
                    'std_visibility': np.std(visibilities),
                    'min_visibility': np.min(visibilities),
                    'max_visibility': np.max(visibilities)
                }
        
        return {
            'total_frames': total_frames,
            'frames_with_pose': frames_with_pose,
            'pose_detection_rate': frames_with_pose / total_frames,
            'landmark_visibilities': landmark_visibilities
        }

def main():
    """Main function to run pose extraction."""
    extractor = PoseExtractor()
    results = extractor.extract_pose_from_all_directories()
    
    # Print summary
    print("\n" + "="*50)
    print("POSE EXTRACTION SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"Frame Directory: {Path(result['frame_dir']).name}")
        print(f"  - Processed {result['total_images']} images")
        print(f"  - Successful pose extractions: {result['successful_extractions']}")
        print(f"  - Detection rate: {result['successful_extractions']/result['total_images']:.2%}")
        print(f"  - Output: {result['output_dir']}")
        print()

if __name__ == "__main__":
    main() 