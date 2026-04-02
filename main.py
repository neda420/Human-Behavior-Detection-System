#!/usr/bin/env python3
"""
Human Behavior Detection and Alert System
=========================================

This is the main application that orchestrates the entire system:
1. Extract frames from videos
2. Extract pose keypoints from frames
3. Classify behaviors using AI
4. Trigger alerts for suspicious behaviors
5. Provide chatbot interface for queries

Usage:
    python main.py --mode [extract_frames|extract_pose|train_model|analyze|chat|full_pipeline|interactive]
"""

import argparse
import logging
import sys
import cv2
import time
import threading
from pathlib import Path
import os
from typing import List, Dict, Optional
import pickle
import json

# Force UTF-8 output on Windows consoles to avoid UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Import our modules
from preprocessing.extract_frames import FrameExtractor
from preprocessing.extract_pose import PoseExtractor
from ai_model.predict_behavior import BehaviorClassifier
from alerts.alert_trigger import AlertTrigger, BehaviorMonitor
from chatbot.chatbot_interface import BehaviorChatbot
from ai_model.cnn_classifier import BehaviorCNNClassifier, TrainConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HumanBehaviorDetectionSystem:
    """Main system class that orchestrates all components."""
    
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.pose_extractor = PoseExtractor()
        self.behavior_classifier = BehaviorClassifier()
        self.cnn_classifier = None
        self.alert_trigger = AlertTrigger()
        self.behavior_monitor = BehaviorMonitor(self.alert_trigger)
        self.chatbot = BehaviorChatbot()
        
        # Create necessary directories
        self.create_directories()
        
        # Real-time processing variables
        self.realtime_active = False
        self.cap = None
        
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            "data/extracted_frames",
            "data/keypoints", 
            "data/labels",
            "data/logs",
            "ai_model",
            "alerts"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def extract_frames_from_videos(self, fps: int = 1) -> List[Dict]:
        """
        Extract frames from all videos in the raw_videos directory.
        
        Args:
            fps: Frames per second to extract
            
        Returns:
            List of extraction results
        """
        logger.info("Starting frame extraction...")
        
        self.frame_extractor.fps = fps
        results = self.frame_extractor.extract_frames_from_directory()
        
        logger.info(f"Frame extraction completed. Processed {len(results)} videos.")
        return results
    
    def extract_pose_from_frames(self) -> List[Dict]:
        """
        Extract pose keypoints from all extracted frames.
        
        Returns:
            List of pose extraction results
        """
        logger.info("Starting pose extraction...")
        
        results = self.pose_extractor.extract_pose_from_all_directories()
        
        logger.info(f"Pose extraction completed. Processed {len(results)} frame directories.")
        return results
    
    def train_behavior_model(self, use_synthetic_data: bool = True) -> bool:
        """
        Train the behavior classification model.
        
        Args:
            use_synthetic_data: Whether to use synthetic data for training
            
        Returns:
            True if training was successful
        """
        logger.info("Starting model training...")
        
        try:
            if use_synthetic_data:
                # Use synthetic data for demonstration
                from ai_model.predict_behavior import create_synthetic_training_data
                synthetic_data = create_synthetic_training_data()
                
                # Prepare training data
                pose_data_dirs = []
                labels = []
                
                for behavior, pose_data in synthetic_data.items():
                    # Save synthetic data
                    output_dir = Path(f"data/keypoints/synthetic_{behavior}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    import pickle
                    with open(output_dir / "pose_data.pkl", 'wb') as f:
                        pickle.dump(pose_data, f)
                    
                    pose_data_dirs.append(str(output_dir))
                    labels.append(behavior)
                
                # Train model
                self.behavior_classifier.train_model(pose_data_dirs, labels)
                logger.info("Model training completed with synthetic data.")
                return True
            else:
                # Use real data (requires labeled data)
                logger.warning("Real data RF training is not used when CNN training is requested.")
                return False
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    def train_cnn_model(self, epochs: int = 5, batch_size: int = 64) -> bool:
        """Train MobileNetV3 on extracted frames using folder names as labels."""
        try:
            frames_root = Path("data/extracted_frames")
            # If no frames, extract from raw videos first
            if not frames_root.exists() or not any(d.is_dir() for d in frames_root.iterdir()):
                logger.info("No extracted frames found. Extracting from raw videos...")
                self.extract_frames_from_videos(fps=1)

            config = TrainConfig(
                frames_root=str(frames_root),
                epochs=epochs,
                batch_size=batch_size,
            )
            self.cnn_classifier = BehaviorCNNClassifier()
            result = self.cnn_classifier.train(config)
            logger.info(f"CNN training complete. Best val acc: {result['best_val_acc']:.3f}")
            return True
        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            return False
    
    def analyze_videos(self) -> Dict:
        """
        Analyze all processed videos for behavior detection.
        
        Returns:
            Analysis results
        """
        logger.info("Starting video analysis...")
        
        # Start monitoring
        self.behavior_monitor.start_monitoring()
        
        analysis_results = {}
        
        # Get all pose data directories
        keypoints_dir = Path("data/keypoints")
        if not keypoints_dir.exists():
            logger.error("No pose data found. Please run pose extraction first.")
            return analysis_results
        
        pose_dirs = [d for d in keypoints_dir.iterdir() if d.is_dir()]
        
        for pose_dir in pose_dirs:
            video_name = pose_dir.name
            pose_data_file = pose_dir / "pose_data.pkl"
            
            if not pose_data_file.exists():
                logger.warning(f"No pose data found for {video_name}")
                continue
            
            logger.info(f"Analyzing {video_name}...")
            
            # Load pose data
            import pickle
            with open(pose_data_file, 'rb') as f:
                pose_data = pickle.load(f)
            
            # Predict behaviors
            predictions = self.behavior_classifier.predict_behavior(pose_data)
            
            if predictions:
                # Get behavior summary
                summary = self.behavior_classifier.get_behavior_summary(predictions)
                
                # Process for alerts
                alerts = self.behavior_monitor.process_behavior_results(summary, video_name)
                if alerts is None:
                    alerts = []
                
                analysis_results[video_name] = {
                    'predictions': predictions,
                    'summary': summary,
                    'alerts': alerts
                }
                
                logger.info(f"Analysis completed for {video_name}")
                logger.info(f"  - Total frames: {summary['total_frames']}")
                logger.info(f"  - Behaviors detected: {list(summary['behaviors'].keys())}")
                logger.info(f"  - Alerts triggered: {len(alerts)}")
        
        # Stop monitoring
        self.behavior_monitor.stop_monitoring()
        
        logger.info(f"Video analysis completed. Processed {len(analysis_results)} videos.")
        return analysis_results
    
    def analyze_saved_video(self, video_path: str) -> Dict:
        """
        Analyze a single saved video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing saved video: {video_path}")
        
        # Check if video exists
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            logger.error(f"Video file not found: {video_path}")
            return {}
        
        # Extract frames from this video
        frame_result = self.frame_extractor.extract_frames_from_video(video_path_obj)
        if not frame_result:
            logger.error(f"Failed to extract frames from {video_path}")
            return {}
        
        # Extract pose from frames
        frame_dir = Path(frame_result['output_dir'])
        pose_result = self.pose_extractor.extract_pose_from_directory(frame_dir)
        if not pose_result:
            logger.error(f"Failed to extract pose from frames")
            return {}
        
        # Load pose data
        pose_data_file = Path(pose_result['pose_data_pickle'])
        import pickle
        with open(pose_data_file, 'rb') as f:
            pose_data = pickle.load(f)
        
        # Predict behaviors
        predictions = self.behavior_classifier.predict_behavior(pose_data)
        
        if predictions:
            # Get behavior summary
            summary = self.behavior_classifier.get_behavior_summary(predictions)
            
            # Process for alerts
            self.behavior_monitor.start_monitoring()
            alerts = self.behavior_monitor.process_behavior_results(summary, video_path_obj.name)
            self.behavior_monitor.stop_monitoring()
            
            # Print results
            if alerts is not None:
                self.print_analysis_results(video_path_obj.name, summary, alerts)
            else:
                self.print_analysis_results(video_path_obj.name, summary, [])
            
            return {
                'video_name': video_path_obj.name,
                'predictions': predictions,
                'summary': summary,
                'alerts': alerts or []
            }
        
        return {}
    
    def start_realtime_analysis(self, camera_index: int = 0):
        """Start real-time behavior analysis from camera.

        Architecture (Phase 7):
        • **Producer thread** continuously captures frames from the camera and
          puts them into a bounded ``queue.Queue``.  If the queue is full the
          oldest frame is dropped so the display stays live.
        • **Consumer (main thread)** pulls frames from the queue, runs pose
          extraction and CNN inference, draws overlays, and shows the result.
        • Inference is *adaptive*: it is skipped when processing the previous
          frame took longer than ``_INFERENCE_BUDGET_S`` seconds, preventing
          the display from lagging behind.
        • A live FPS counter is drawn in the top-right corner.
        """
        import queue as _queue

        logger.info(f"Starting real-time analysis from camera {camera_index}")

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return

        cam_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera: {width}×{height} @ {cam_fps:.1f} fps")

        session_name = f"realtime_session_{time.strftime('%Y%m%d_%H%M%S')}"
        session_dir  = Path("data/keypoints") / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        self.behavior_monitor.start_monitoring()
        self.realtime_active = True

        # ── Shared state ────────────────────────────────────────────────────
        frame_queue: _queue.Queue = _queue.Queue(maxsize=4)

        pose_session_data: list = []
        session_predictions: dict = {
            'predictions': [], 'probabilities': [], 'confidence_scores': []
        }
        current_overlay: Optional[Dict] = None

        # FPS tracking
        fps_counter    = 0
        fps_display    = 0.0
        fps_last_time  = time.time()

        _INFERENCE_BUDGET_S = 0.08   # 80 ms → up to 12 Hz inference max

        # ── Producer thread ──────────────────────────────────────────────────
        def _producer():
            while self.realtime_active:
                ret, frm = self.cap.read()
                if not ret:
                    logger.warning("Camera read failed in producer thread")
                    break
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()   # drop oldest frame
                    except _queue.Empty:
                        pass
                frame_queue.put(frm)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        # Ensure CNN is ready
        if self.cnn_classifier is None:
            try:
                self.cnn_classifier = BehaviorCNNClassifier()
                if self.cnn_classifier.model is None:
                    self.cnn_classifier = None
            except Exception:
                self.cnn_classifier = None

        print("\n🎥 Real-time Behavior Detection Started")
        print("Press 'q' to quit, 'c' to chat with chatbot")
        print("=" * 50)

        try:
            while self.realtime_active:
                # ── Get latest frame from queue ──────────────────────────────
                try:
                    frame = frame_queue.get(timeout=0.5)
                except _queue.Empty:
                    continue

                t_frame_start = time.time()

                # ── Pose extraction (every frame for overlay) ────────────────
                pose_result = self.pose_extractor.extract_pose_from_ndarray(frame)
                if pose_result:
                    pose_session_data.append(pose_result)

                # ── CNN inference (adaptive — skip if over budget) ───────────
                elapsed_since_last = time.time() - t_frame_start
                if elapsed_since_last < _INFERENCE_BUDGET_S and self.cnn_classifier is not None:
                    try:
                        pred, conf, probs_dict = self.cnn_classifier.predict_ndarray(frame)
                        if pred is not None:
                            ts  = time.strftime("%H:%M:%S")
                            top3 = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                            top3_str = ", ".join(f"{k}:{v:.2f}" for k, v in top3)
                            print(f"[{ts}] Behavior: {pred} (Conf: {conf:.2f}) | {top3_str}")

                            session_predictions['predictions'].append(str(pred))
                            session_predictions['confidence_scores'].append(float(conf))
                            session_predictions['probabilities'].append(
                                float(max(probs_dict.values())) if probs_dict else float(conf))

                            v = max(0.0, min(1.0, (
                                1.0 * probs_dict.get('fighting', 0.0)
                                + 1.0 * probs_dict.get('assault_violence', 0.0)
                                + 1.0 * probs_dict.get('violence', 0.0)
                                + 0.8 * probs_dict.get('falling', 0.0)
                                + 0.3 * probs_dict.get('running', 0.0)
                            )))
                            current_overlay = {
                                'label': str(pred), 'confidence': float(conf),
                                'probs': probs_dict, 'violence': v,
                            }
                            if (pose_result and pose_result.get('landmark_coordinates')
                                    and 'nose' in pose_result['landmark_coordinates']):
                                nose_lm = pose_result['landmark_coordinates']['nose']
                                current_overlay['nose_norm'] = (float(nose_lm['x']), float(nose_lm['y']))

                            alerts = self.behavior_monitor.process_behavior_results(
                                {'behavior_percentages': probs_dict}, "realtime_camera")
                            if alerts:
                                print(f"[{ts}] 🚨 ALERT: {alerts[0]['behavior']} detected!")
                    except Exception as e:
                        logger.warning(f"CNN inference failed: {e}")

                # ── Draw overlays ────────────────────────────────────────────
                if pose_result and pose_result.get('pose_detected'):
                    self.draw_pose_on_frame(frame, pose_result)
                if current_overlay is not None:
                    self.draw_behavior_overlay(frame, current_overlay)
                    self.draw_head_overlay(frame, current_overlay)

                # ── FPS counter ──────────────────────────────────────────────
                fps_counter += 1
                now = time.time()
                if now - fps_last_time >= 1.0:
                    fps_display   = fps_counter / (now - fps_last_time)
                    fps_counter   = 0
                    fps_last_time = now
                fps_text = f"FPS: {fps_display:.1f}"
                h_f, w_f = frame.shape[:2]
                (tw, _th), _base = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(frame, fps_text, (w_f - tw - 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow('Real-time Behavior Detection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.start_chatbot_session()

        except KeyboardInterrupt:
            print("\n⚠️  Real-time analysis interrupted")
        finally:
            self.realtime_active = False
            producer_thread.join(timeout=2.0)
            try:
                with open(session_dir / "pose_data.pkl", 'wb') as f:
                    pickle.dump(pose_session_data, f)
                session_pred_dict = {
                    'predictions':     session_predictions['predictions'],
                    'probabilities':   session_predictions['probabilities'],
                    'confidence_scores': session_predictions['confidence_scores']
                    or [0.0] * len(session_predictions['predictions']),
                }
                session_summary = self.behavior_classifier.get_behavior_summary(session_pred_dict)
                with open(session_dir / "summary.json", 'w') as f:
                    json.dump(session_summary, f, indent=2)
                print(f"\n📝 Saved real-time session data to: {session_dir}")
            except Exception as e:
                logger.error(f"Failed to save real-time session data: {e}")
            finally:
                self.stop_realtime_analysis()
    
    def stop_realtime_analysis(self):
        """Stop real-time analysis."""
        self.realtime_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.behavior_monitor.stop_monitoring()
        logger.info("Real-time analysis stopped")
    
    def draw_pose_on_frame(self, frame, pose_result):
        """Draw pose landmarks on frame."""
        if not pose_result['pose_detected'] or not pose_result['landmark_coordinates']:
            return
        
        landmarks = pose_result['landmark_coordinates']
        height, width = frame.shape[:2]
        
        # Draw key landmarks
        key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
                        'right_knee', 'left_ankle', 'right_ankle']
        
        for landmark_name in key_landmarks:
            if landmark_name in landmarks:
                landmark = landmarks[landmark_name]
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)
                
                # Draw circle for landmark
                if landmark_name == 'nose':
                    cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)  # red nose highlight
                    cv2.putText(frame, 'nose', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Draw landmark name
                if landmark_name != 'nose':
                    cv2.putText(frame, landmark_name, (x + 10, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def draw_behavior_overlay(self, frame, overlay: Dict):
        """Draw behavior meters and labels on the frame."""
        h, w = frame.shape[:2]
        x0, y0 = 10, 25
        label = overlay.get('label', 'unknown')
        conf = overlay.get('confidence', 0.0)
        cv2.putText(frame, f"Behavior: {label}  ({conf:.2f})", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y = y0 + 15
        probs = overlay.get('probs', {})
        normal_p = float(probs.get('normal', 0.0))
        # Combine all violent labels into the fighting meter for display
        fighting_p = float(
            probs.get('fighting', 0.0)
            + probs.get('assault_violence', 0.0)
            + probs.get('violence', 0.0)
        )
        running_p = float(probs.get('running', 0.0))
        violence = float(overlay.get('violence', 0.0))
        meters = [
            ('Normal', normal_p, (0, 200, 0)),
            ('Fighting', fighting_p, (0, 0, 255)),
            ('Running', running_p, (255, 140, 0)),
            ('Violence', violence, (0, 0, 200)),
        ]
        bar_w, bar_h, spacing = 220, 12, 8
        y += 20
        for name, val, color in meters:
            # background
            cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), -1)
            # fill
            fill_w = int(bar_w * max(0.0, min(1.0, val)))
            cv2.rectangle(frame, (x0, y), (x0 + fill_w, y + bar_h), color, -1)
            # label
            cv2.putText(frame, f"{name}: {val:.2f}", (x0 + bar_w + 10, y + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y += bar_h + spacing

    def draw_head_overlay(self, frame, overlay: Dict):
        
        nose_norm = overlay.get('nose_norm')
        if not nose_norm:
            return
        h, w = frame.shape[:2]
        nx = int(max(0, min(1, nose_norm[0])) * w)
        ny = int(max(0, min(1, nose_norm[1])) * h)
        violence = float(overlay.get('violence', 0.0))
        label = str(overlay.get('label', 'unknown'))
        conf = float(overlay.get('confidence', 0.0))
        # Color scale by violence
        if violence < 0.2:
            color = (0, 200, 0)
        elif violence < 0.5:
            color = (0, 200, 200)
        elif violence < 0.8:
            color = (0, 140, 255)
        else:
            color = (0, 0, 255)
        radius = 12 + int(violence * 18)
        cv2.circle(frame, (nx, ny), radius, color, 2)
        cv2.putText(frame, "V", (nx - 6, ny + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (nx - 20, ny - radius - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    
    def print_analysis_results(self, video_name: str, summary: Dict, alerts: List):
        """Print analysis results in a formatted way."""
        print(f"\n📊 ANALYSIS RESULTS: {video_name}")
        print("=" * 50)
        print(f"📹 Video: {video_name}")
        print(f"📊 Total frames: {summary['total_frames']}")
        print(f"🎯 High confidence frames: {summary['high_confidence_frames']}")
        print(f"📈 Confidence rate: {summary['confidence_rate']:.1%}")
        
        if summary['behaviors']:
            print(f"\n🎭 Detected Behaviors:")
            for behavior, count in summary['behaviors'].items():
                percentage = summary['behavior_percentages'][behavior]
                print(f"   • {behavior}: {count} frames ({percentage:.1%})")
        
        if alerts:
            print(f"\n🚨 Alerts Triggered: {len(alerts)}")
            for alert in alerts:
                print(f"   • {alert['behavior']} - {alert['severity']} severity")
        
        print("=" * 50)
    
    def start_chatbot_session(self):
        """Start an interactive chatbot session."""
        print("\n🤖 Starting chatbot session...")
        print("Type 'exit' to return to main menu")
        print("-" * 30)
        # Refresh chatbot data so it reflects latest sessions and alerts
        try:
            self.chatbot.load_data()
        except Exception:
            pass
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['exit', 'quit', 'back']:
                    print("🤖 Returning to main menu...")
                    break
                
                if not query:
                    continue
                
                response = self.chatbot.process_query(query)
                print(f"🤖 {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n🤖 Returning to main menu...")
                break
            except Exception as e:
                print(f"🤖 Error: {e}")
    
    def run_full_pipeline(self, fps: int = 1, use_synthetic_data: bool = True) -> Dict:
        """
        Run the complete pipeline from video to analysis.
        
        Args:
            fps: Frames per second to extract
            use_synthetic_data: Whether to use synthetic data for training
            
        Returns:
            Pipeline results
        """
        logger.info("Starting full pipeline...")
        
        results = {
            'frame_extraction': [],
            'pose_extraction': [],
            'model_training': False,
            'analysis': {}
        }
        
        try:
            # Step 1: Extract frames
            logger.info("Step 1: Extracting frames...")
            results['frame_extraction'] = self.extract_frames_from_videos(fps)
            
            # Step 2: Extract pose keypoints
            logger.info("Step 2: Extracting pose keypoints...")
            results['pose_extraction'] = self.extract_pose_from_frames()
            
            # Step 3: Train model
            logger.info("Step 3: Training behavior model...")
            results['model_training'] = self.train_behavior_model(use_synthetic_data)
            
            # Step 4: Analyze videos
            logger.info("Step 4: Analyzing videos...")
            results['analysis'] = self.analyze_videos()
            
            logger.info("Full pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
            results['error'] = str(e)
        
        return results
    
    def show_interactive_menu(self):
        """Show interactive menu for user selection."""
        while True:
            print("\n" + "=" * 60)
            print("🤖 HUMAN BEHAVIOR DETECTION SYSTEM")
            print("=" * 60)
            print("1. 📹 Analyze Saved Videos")
            print("2. 🎥 Real-time Camera Analysis")
            print("3. 🧠 Train CNN (MobileNetV3)")
            print("9. 🔧 Calibrate CNN (Temperature)")
            print("0. 🚪 Exit")
            print("=" * 60)
            
            try:
                choice = input("Select an option (0-3): ").strip()
                
                if choice == '0':
                    print("👋 Goodbye!")
                    break
                elif choice == '1':
                    self.handle_saved_video_analysis()
                elif choice == '2':
                    self.handle_realtime_analysis()
                elif choice == '3':
                    self.handle_train_cnn()
                elif choice == '9':
                    self.handle_calibrate_cnn()
                else:
                    print("❌ Invalid option. Please select 0-3.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def handle_saved_video_analysis(self):
        """Handle saved video analysis option."""
        print("\n📹 SAVED VIDEO ANALYSIS")
        print("-" * 30)
        
        # Check for videos in raw_videos directory
        raw_videos_dir = Path("data/raw_videos")
        video_files = list(raw_videos_dir.glob("*.mp4"))
        
        if not video_files:
            print("❌ No MP4 videos found in data/raw_videos/")
            print("Please add some videos first.")
            return
        
        print(f"Found {len(video_files)} videos:")
        for i, video_file in enumerate(video_files, 1):
            print(f"{i}. {video_file.name}")
        
        try:
            choice = input(f"\nSelect video (1-{len(video_files)}) or 'all' for all videos: ").strip()
            
            if choice.lower() == 'all':
                print("🔄 Analyzing all videos...")
                for video_file in video_files:
                    self.analyze_saved_video(str(video_file))
            else:
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(video_files):
                        selected_video = video_files[index]
                        self.analyze_saved_video(str(selected_video))
                    else:
                        print("❌ Invalid selection.")
                except ValueError:
                    print("❌ Invalid input.")
                    
        except KeyboardInterrupt:
            print("\n⚠️  Analysis cancelled")
    
    def handle_realtime_analysis(self):
        """Handle real-time analysis option."""
        print("\n🎥 REAL-TIME CAMERA ANALYSIS")
        print("-" * 30)
        
        # Require CNN model trained on your dataset
        cnn_model_file = Path("ai_model/mobilenet_v3_behavior.pth")
        if not cnn_model_file.exists():
            print("❌ CNN model not found. Please train it first (option 3: Train CNN).")
            return
        
        # Auto-start default camera without prompting
        try:
            camera_index = 0
            print(f"🎥 Starting real-time analysis from camera {camera_index}...")
            print("Press 'q' to quit, 'c' to chat with chatbot")
            self.start_realtime_analysis(camera_index)

            # After quitting real-time, automatically open chatbot with refreshed data
            print("\n💬 Opening chatbot with latest session data...")
            self.start_chatbot_session()
        except KeyboardInterrupt:
            print("\n⚠️  Real-time analysis cancelled")
    
    def handle_full_pipeline(self):
        """Handle full pipeline option."""
        print("\n🔄 FULL PIPELINE")
        print("-" * 30)
        
        try:
            fps_choice = input("Frames per second to extract (0 = all frames, default: 1): ").strip()
            fps = int(fps_choice) if fps_choice else 1
            
            print(f"🔄 Running full pipeline with {fps} fps...")
            results = self.run_full_pipeline(fps=fps, use_synthetic_data=True)
            
            if 'error' not in results:
                print("✅ Full pipeline completed successfully!")
            else:
                print(f"❌ Pipeline failed: {results['error']}")
                
        except KeyboardInterrupt:
            print("\n⚠️  Pipeline cancelled")
        except ValueError:
            print("❌ Invalid fps value")

    def handle_train_cnn(self):
        """Handle CNN training option."""
        print("\n🧠 TRAIN CNN (MobileNetV3)")
        print("-" * 30)
        try:
            ckpt_path = Path("ai_model/mobilenet_v3_behavior.pth")
            meta_path = Path("ai_model/mobilenet_v3_behavior.meta.json")

            if ckpt_path.exists():
                choice = input("Use existing model without training [keep], resume training [resume], or start fresh [fresh]? (default: keep): ").strip().lower()
                if choice in ("", "keep", "k"):
                    print("ℹ️ Keeping existing model. No training performed.")
                    return
                elif choice in ("fresh", "f"):
                    # Delete existing checkpoints for a fresh start
                    try:
                        if ckpt_path.exists():
                            os.remove(ckpt_path)
                        if meta_path.exists():
                            os.remove(meta_path)
                        print("🧹 Removed previous CNN checkpoints. Starting fresh training.")
                    except Exception as e:
                        print(f"⚠️ Could not remove previous checkpoints: {e}")
                elif choice not in ("resume", "r"):
                    print("❌ Invalid choice. Aborting.")
                    return

            epochs_in = input("Epochs (default 10): ").strip()
            batch_in = input("Batch size (default 64): ").strip()
            epochs = int(epochs_in) if epochs_in else 10
            batch = int(batch_in) if batch_in else 64
            print(f"🚀 Training CNN for {epochs} epochs, batch size {batch}...")
            ok = self.train_cnn_model(epochs=epochs, batch_size=batch)
            if ok:
                print("✅ CNN training complete.")
            else:
                print("❌ CNN training failed.")
        except KeyboardInterrupt:
            print("\n⚠️  CNN training cancelled")
        except ValueError:
            print("❌ Invalid numeric input")

    def handle_calibrate_cnn(self):
        """Calibrate CNN temperature from frames and save to meta."""
        print("\n🔧 CALIBRATE CNN (Temperature)")
        print("-" * 30)
        try:
            frames_root = str(Path("data/extracted_frames").resolve())
            if not Path(frames_root).exists():
                print("❌ No extracted frames found. Please extract frames or train first.")
                return
            # Ensure model is loaded
            if self.cnn_classifier is None:
                self.cnn_classifier = BehaviorCNNClassifier()
            T = self.cnn_classifier.calibrate_from_frames(frames_root=frames_root)
            print(f"✅ Calibrated temperature saved: T={T:.3f}")
        except KeyboardInterrupt:
            print("\n⚠️  Calibration cancelled")
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
    
    def run_system_test(self):
        """Run system test."""
        print("\n🧪 SYSTEM TEST")
        print("-" * 30)
        
        try:
            import test_system
            test_system.run_full_test()
        except ImportError:
            print("❌ Test system not available")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def print_system_status(self):
        """Print the current status of the system."""
        print("\n" + "="*60)
        print("📊 SYSTEM STATUS")
        print("="*60)
        
        # Check raw videos
        raw_videos_dir = Path("data/raw_videos")
        if raw_videos_dir.exists():
            video_files = list(raw_videos_dir.glob("*.mp4"))
            print(f"📹 Raw videos: {len(video_files)} found")
        else:
            print("📹 Raw videos: No directory found")
        
        # Check extracted frames
        frames_dir = Path("data/extracted_frames")
        if frames_dir.exists():
            frame_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
            print(f"🖼️  Extracted frames: {len(frame_dirs)} video directories")
        else:
            print("🖼️  Extracted frames: No directory found")
        
        # Check pose data
        keypoints_dir = Path("data/keypoints")
        if keypoints_dir.exists():
            pose_dirs = [d for d in keypoints_dir.iterdir() if d.is_dir()]
            print(f"🎭 Pose data: {len(pose_dirs)} directories")
        else:
            print("🎭 Pose data: No directory found")
        
        # Check trained model
        model_file = Path("ai_model/behavior_classifier.pkl")
        if model_file.exists():
            print("🤖 Behavior model: Trained and ready")
        else:
            print("🤖 Behavior model: Not trained")
        
        # Check alerts
        alert_log = Path("data/logs/alerts.log")
        if alert_log.exists():
            with open(alert_log, 'r') as f:
                alert_count = len(f.readlines())
            print(f"🚨 Alerts: {alert_count} logged")
        else:
            print("🚨 Alerts: No alerts logged")
        
        print("="*60)

def main():
    """Main function to handle command line arguments and run the system."""
    parser = argparse.ArgumentParser(
        description="Human Behavior Detection and Alert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode interactive    # Start interactive menu
  python main.py --mode full_pipeline  # Run complete pipeline
  python main.py --mode extract_frames # Extract frames only
  python main.py --mode extract_pose   # Extract pose only
  python main.py --mode analyze        # Analyze videos only
  python main.py --mode chat           # Start chatbot
  python main.py --status              # Show system status
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['extract_frames', 'extract_pose', 'train_model', 'train_cnn', 'analyze', 'chat', 'full_pipeline', 'interactive'],
        default='interactive',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second to extract (default: 1)'
    )
    
    parser.add_argument(
        '--no-synthetic',
        action='store_true',
        help='Do not use synthetic data for training'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = HumanBehaviorDetectionSystem()
    
    # Show status if requested
    if args.status:
        system.print_system_status()
        return
    
    # Run based on mode
    try:
        if args.mode == 'interactive':
            system.show_interactive_menu()
        elif args.mode == 'extract_frames':
            system.extract_frames_from_videos(args.fps)
        elif args.mode == 'extract_pose':
            system.extract_pose_from_frames()
        elif args.mode == 'train_model':
            system.train_behavior_model(not args.no_synthetic)
        elif args.mode == 'train_cnn':
            system.handle_train_cnn()
        elif args.mode == 'analyze':
            system.analyze_videos()
        elif args.mode == 'chat':
            system.start_chatbot_session()
        elif args.mode == 'full_pipeline':
            system.run_full_pipeline(args.fps, not args.no_synthetic)
        
        # Show final status (except for interactive mode)
        if args.mode != 'interactive':
            system.print_system_status()
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 