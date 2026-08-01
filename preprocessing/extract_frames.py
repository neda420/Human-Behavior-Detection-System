import cv2
import os
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, input_dir="data/raw_videos", output_dir="data/extracted_frames", fps=1):
        """
        Initialize the frame extractor.
        
        Args:
            input_dir (str): Directory containing input videos
            output_dir (str): Directory to save extracted frames
            fps (int): Frames per second to extract (1 = 1 frame per second)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames_from_video(self, video_path, output_subdir=None):
        """
        Extract frames from a single video file.
        
        Args:
            video_path (str): Path to the video file
            output_subdir (str): Subdirectory name for output frames
            
        Returns:
            dict: Information about extracted frames
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
            
        # Create output subdirectory
        if output_subdir is None:
            output_subdir = video_path.stem
            
        frame_output_dir = self.output_dir / output_subdir
        frame_output_dir.mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0:
            video_fps = float(self.fps)
        duration = total_frames / max(video_fps, 1.0)
        
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"Total frames: {total_frames}, FPS: {video_fps:.2f}, Duration: {duration:.2f}s")
        
        # Calculate frame interval
        # If fps <= 0, extract every frame
        if self.fps and self.fps > 0:
            frame_interval = max(int(video_fps / max(self.fps, 1)), 1)
        else:
            frame_interval = 1
        
        frame_count = 0
        saved_count = 0
        
        # Extract frames
        with tqdm(total=total_frames, desc=f"Extracting frames from {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Save frame at specified interval
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{saved_count:06d}.jpg"
                    frame_path = frame_output_dir / frame_filename
                    
                    # Save frame
                    cv2.imwrite(str(frame_path), frame)
                    saved_count += 1
                    
                frame_count += 1
                pbar.update(1)
                
        cap.release()
        
        logger.info(f"Extracted {saved_count} frames from {video_path.name}")
        
        return {
            'video_path': str(video_path),
            'output_dir': str(frame_output_dir),
            'total_frames': total_frames,
            'extracted_frames': saved_count,
            'video_fps': video_fps,
            'extraction_fps': self.fps if self.fps > 0 else 'all'
        }
        
    def extract_frames_from_directory(self):
        """
        Extract frames from all video files in the input directory.
        
        Returns:
            list: List of extraction results for each video
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []
        
        # Find all video files
        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(f"*{ext}"))
            
        if not video_files:
            logger.warning(f"No video files found in {self.input_dir}")
            return []
            
        logger.info(f"Found {len(video_files)} video files")
        
        results = []
        for video_file in video_files:
            result = self.extract_frames_from_video(video_file)
            if result:
                results.append(result)
                
        return results
        
    def get_frame_info(self, frame_dir):
        """
        Get information about extracted frames in a directory.
        
        Args:
            frame_dir (str): Directory containing frames
            
        Returns:
            dict: Frame information
        """
        frame_dir = Path(frame_dir)
        
        if not frame_dir.exists():
            return None
            
        frame_files = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))
        
        return {
            'frame_dir': str(frame_dir),
            'total_frames': len(frame_files),
            'frame_files': [str(f) for f in sorted(frame_files)]
        }

def main():
    """Main function to run frame extraction."""
    extractor = FrameExtractor()
    results = extractor.extract_frames_from_directory()
    
    # Print summary
    print("\n" + "="*50)
    print("FRAME EXTRACTION SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"Video: {Path(result['video_path']).name}")
        print(f"  - Extracted {result['extracted_frames']} frames")
        print(f"  - Output: {result['output_dir']}")
        print()

if __name__ == "__main__":
    main() 
