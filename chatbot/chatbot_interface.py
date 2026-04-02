import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import – avoids circular dependencies and heavy import at module load time.
def _get_behavior_classifier():
    from ai_model.predict_behavior import BehaviorClassifier
    return BehaviorClassifier

class BehaviorChatbot:
    """Chatbot interface for human behavior detection system."""

    _MODEL_PATH = "ai_model/behavior_classifier.pkl"

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.behavior_data = {}
        self.alert_data = {}
        self.video_metadata = {}
        self.realtime_summary: Dict[str, Any] = {}
        # ML classifier + prediction cache (video_name → behavior summary dict)
        self._classifier = None
        self._prediction_cache: Dict[str, Dict] = {}
        self._load_classifier()
        self.load_data()
        
        # Define response templates
        self.response_templates = {
            'greeting': [
                "Hello! I'm your behavior detection assistant. How can I help you today?",
                "Hi there! I can help you understand what's happening in your videos. What would you like to know?",
                "Welcome! I'm here to answer questions about detected behaviors and video analysis."
            ],
            'no_data': [
                "I don't have any behavior data available yet. Please run the video analysis first.",
                "No behavior detection results found. Try processing some videos first.",
                "I need to analyze some videos before I can answer questions about behaviors."
            ],
            'unknown_query': [
                "I'm not sure I understand. Could you rephrase your question?",
                "I don't have information about that. Try asking about detected behaviors, alerts, or video analysis.",
                "I'm still learning! Try asking about what behaviors were detected or what alerts were triggered."
            ]
        }
    
    def _load_classifier(self):
        """Load the trained BehaviorClassifier (silently skips if not found)."""
        try:
            BehaviorClassifier = _get_behavior_classifier()
            clf = BehaviorClassifier(model_path=self._MODEL_PATH)
            if clf.model is not None:
                self._classifier = clf
                logger.info("BehaviorClassifier loaded into chatbot.")
            else:
                logger.warning("BehaviorClassifier model not trained yet; chatbot will show raw pose stats.")
        except Exception as e:
            logger.warning(f"Could not load BehaviorClassifier: {e}")

    def _classify_pose_data(self, video_name: str) -> Dict[str, int]:
        """Return a behavior-count dict for *video_name* using the ML classifier.

        Results are cached to avoid re-running inference on every chatbot query.
        Falls back to an empty dict if the classifier is unavailable.
        """
        if video_name in self._prediction_cache:
            return self._prediction_cache[video_name]

        pose_data = self.behavior_data.get(video_name)
        if not pose_data:
            return {}

        if self._classifier is None:
            return {}

        try:
            predictions = self._classifier.predict_with_smoothing(pose_data)
            if predictions is None or not len(predictions.get('predictions', [])):
                return {}
            summary = self._classifier.get_behavior_summary(predictions)
            behavior_counts = dict(summary.get('behaviors', {}))
            self._prediction_cache[video_name] = behavior_counts
            return behavior_counts
        except Exception as e:
            logger.error(f"Classifier inference failed for {video_name}: {e}")
            return {}

    def load_data(self):
        """Load behavior detection data and alerts."""
        # Invalidate prediction cache on reload
        self._prediction_cache.clear()
        try:
            # Load behavior data from keypoints directory
            keypoints_dir = self.data_dir / "keypoints"
            if keypoints_dir.exists():
                for subdir in keypoints_dir.iterdir():
                    if subdir.is_dir():
                        pose_data_file = subdir / "pose_data.pkl"
                        if pose_data_file.exists():
                            with open(pose_data_file, 'rb') as f:
                                self.behavior_data[subdir.name] = pickle.load(f)

            # Load alert data
            alert_log_file = self.data_dir / "logs" / "alerts.log"
            if alert_log_file.exists():
                self.alert_data = self.load_alert_log(alert_log_file)

            # Load video metadata
            self.load_video_metadata()
            # Load latest realtime session summary if available
            self.load_latest_realtime_summary()

            logger.info(f"Loaded data for {len(self.behavior_data)} videos")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
    def load_latest_realtime_summary(self):
        """Load the most recent realtime session summary for chatbot answers."""
        try:
            keypoints_dir = self.data_dir / "keypoints"
            if not keypoints_dir.exists():
                return
            # Find latest realtime_session_* dir
            sessions = sorted([
                d for d in keypoints_dir.iterdir()
                if d.is_dir() and d.name.startswith("realtime_session_")
            ], key=lambda p: p.name)
            if not sessions:
                return
            latest = sessions[-1]
            summary_path = latest / "summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    self.realtime_summary = json.load(f)
                self.realtime_summary["session_name"] = latest.name
        except Exception as e:
            logger.error(f"Failed to load realtime summary: {e}")
    
    def load_alert_log(self, log_file: Path) -> List[Dict]:
        """Load alert log file."""
        alerts = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        alert = json.loads(line.strip())
                        alerts.append(alert)
        except Exception as e:
            logger.error(f"Error loading alert log: {e}")
        
        return alerts
    
    def load_video_metadata(self):
        """Load video metadata from raw videos directory."""
        raw_videos_dir = self.data_dir / "raw_videos"
        if raw_videos_dir.exists():
            for video_file in raw_videos_dir.glob("*.mp4"):
                self.video_metadata[video_file.stem] = {
                    'filename': video_file.name,
                    'path': str(video_file),
                    'size': video_file.stat().st_size,
                    'modified': datetime.fromtimestamp(video_file.stat().st_mtime)
                }
    
    def process_query(self, query: str) -> str:
        """
        Process user query and return appropriate response.
        
        Args:
            query: User's question
            
        Returns:
            Response string
        """
        query = query.lower().strip()
        
        # Check for greetings
        if any(word in query for word in ['hello', 'hi', 'hey', 'greetings']):
            return self.get_random_response('greeting')
        
        # Check for behavior-related queries
        if any(word in query for word in ['behavior', 'behaviors', 'detected', 'found']):
            return self.handle_behavior_query(query)
        
        # Check for alert-related queries
        if any(word in query for word in ['alert', 'alerts', 'suspicious', 'warning']):
            return self.handle_alert_query(query)
        
        # Check for video-related queries
        if any(word in query for word in ['video', 'videos', 'processed', 'analysis']):
            return self.handle_video_query(query)
        
        # Realtime specific queries
        if any(word in query for word in ['realtime', 'live', 'camera', 'now']):
            return self.handle_realtime_query(query)
        
        # Check for specific behavior types
        behavior_types = ['falling', 'fighting', 'loitering', 'running', 'walking', 'normal']
        for behavior in behavior_types:
            if behavior in query:
                return self.handle_specific_behavior_query(query, behavior)
        
        # Check for temporal queries
        if any(word in query for word in ['when', 'time', 'moment', 'during']):
            return self.handle_temporal_query(query)
        
        # Check for statistical queries
        if any(word in query for word in ['how many', 'count', 'percentage', 'statistics']):
            return self.handle_statistical_query(query)
        
        return self.get_random_response('unknown_query')
    
    def handle_behavior_query(self, query: str) -> str:
        """Handle queries about detected behaviors."""
        if not self.behavior_data:
            return self.get_random_response('no_data')
        
        # Extract video name if mentioned
        video_name = self.extract_video_name(query)
        
        if video_name and video_name in self.behavior_data:
            return self.get_video_behavior_summary(video_name)
        else:
            return self.get_overall_behavior_summary()
    
    def handle_alert_query(self, query: str) -> str:
        """Handle queries about alerts."""
        if not self.alert_data:
            return "No alerts have been triggered yet."
        
        # Check for specific alert types
        if 'recent' in query or 'latest' in query:
            return self.get_recent_alerts()
        elif 'critical' in query or 'high' in query:
            return self.get_alerts_by_severity('high')
        else:
            return self.get_alert_summary()

    def handle_realtime_query(self, query: str) -> str:
        """Answer questions about the latest realtime session results."""
        if not self.realtime_summary:
            return "No realtime session summary available yet."
        summary = self.realtime_summary
        lines = [
            f"📡 Latest realtime session: {summary.get('session_name', 'unknown')}",
            f"📊 Total frames: {summary.get('total_frames', 0)}",
            f"🎯 High confidence frames: {summary.get('high_confidence_frames', 0)}",
            f"📈 Confidence rate: {summary.get('confidence_rate', 0.0):.1%}",
        ]
        behaviors = summary.get('behaviors', {})
        if behaviors:
            lines.append("🎭 Behavior distribution:")
            for k, v in behaviors.items():
                pct = summary.get('behavior_percentages', {}).get(k, 0.0)
                lines.append(f"• {k}: {v} ({pct:.1%})")
        return "\n".join(lines)
    
    def handle_video_query(self, query: str) -> str:
        """Handle queries about videos."""
        if not self.video_metadata:
            return "No video metadata available."
        
        if 'processed' in query or 'analyzed' in query:
            return self.get_processed_videos_summary()
        else:
            return self.get_video_list()
    
    def handle_specific_behavior_query(self, query: str, behavior: str) -> str:
        """Handle queries about specific behavior types."""
        if not self.behavior_data:
            return self.get_random_response('no_data')
        
        return self.get_behavior_specific_summary(behavior)
    
    def handle_temporal_query(self, query: str) -> str:
        """Handle temporal queries about when behaviors occurred."""
        if not self.behavior_data:
            return self.get_random_response('no_data')
        
        # Extract time information from query
        time_info = self.extract_time_info(query)
        return self.get_temporal_behavior_summary(time_info)
    
    def handle_statistical_query(self, query: str) -> str:
        """Handle statistical queries."""
        if not self.behavior_data:
            return self.get_random_response('no_data')
        
        return self.get_statistical_summary()
    
    def extract_video_name(self, query: str) -> Optional[str]:
        """Extract video name from query."""
        # Look for video names in the data
        for video_name in self.behavior_data.keys():
            if video_name.lower() in query:
                return video_name
        
        return None
    
    def extract_time_info(self, query: str) -> Dict[str, Any]:
        """Extract time information from query."""
        time_info = {}
        
        # Look for time patterns
        if 'recent' in query or 'latest' in query:
            time_info['period'] = 'recent'
        elif 'earlier' in query or 'before' in query:
            time_info['period'] = 'earlier'
        
        return time_info
    
    def get_video_behavior_summary(self, video_name: str) -> str:
        """Get behavior summary for a specific video using the ML classifier."""
        pose_data = self.behavior_data[video_name]
        total_frames = len(pose_data)
        frames_with_pose = sum(1 for d in pose_data if d['pose_detected'])

        behaviors = self._classify_pose_data(video_name)

        out = f"📹 **Video: {video_name}**\n\n"
        out += "📊 **Analysis Summary:**\n"
        out += f"• Total frames: {total_frames}\n"
        out += f"• Frames with pose detected: {frames_with_pose}\n"
        if total_frames:
            out += f"• Pose detection rate: {frames_with_pose/total_frames:.1%}\n\n"

        if behaviors:
            out += "🎭 **Detected Behaviors (ML classifier):**\n"
            for behavior, count in sorted(behaviors.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_frames if total_frames else 0
                out += f"• {behavior}: {count} frames ({pct:.1%})\n"
        elif self._classifier is None:
            out += "_Classifier not loaded — train a model first (option 3 in the menu)._\n"
        else:
            out += "_No high-confidence behaviors detected in this video._\n"

        return out

    def get_overall_behavior_summary(self) -> str:
        """Get overall behavior summary across all videos using the ML classifier."""
        if not self.behavior_data:
            return self.get_random_response('no_data')

        out = "📊 **Overall Behavior Summary**\n\n"
        out += f"📹 **Videos Analyzed:** {len(self.behavior_data)}\n\n"

        total_frames = 0
        total_poses  = 0
        all_behaviors: Dict[str, int] = {}

        for video_name, pose_data in self.behavior_data.items():
            total_frames += len(pose_data)
            total_poses  += sum(1 for d in pose_data if d['pose_detected'])
            for beh, cnt in self._classify_pose_data(video_name).items():
                all_behaviors[beh] = all_behaviors.get(beh, 0) + cnt

        out += "📈 **Statistics:**\n"
        out += f"• Total frames: {total_frames}\n"
        out += f"• Total poses detected: {total_poses}\n"
        if total_frames:
            out += f"• Overall pose detection rate: {total_poses/total_frames:.1%}\n\n"

        if all_behaviors:
            out += "🎭 **Behavior Distribution (ML classifier):**\n"
            for beh, cnt in sorted(all_behaviors.items(), key=lambda x: x[1], reverse=True):
                pct = cnt / total_frames if total_frames else 0
                out += f"• {beh}: {cnt} frames ({pct:.1%})\n"
        elif self._classifier is None:
            out += "_Classifier not loaded — train a model first (option 3 in the menu)._\n"

        return out

    def get_recent_alerts(self) -> str:
        """Get recent alerts summary."""
        if not self.alert_data:
            return "No alerts found."

        recent_alerts = self.alert_data[-5:]  # Last 5 alerts

        summary = "🚨 **Recent Alerts**\n\n"
        for alert in recent_alerts:
            timestamp = alert.get('timestamp', 'Unknown')
            behavior = alert.get('behavior', 'Unknown')
            level = alert.get('level', 'Unknown')
            video = alert.get('video', 'Unknown')
            
            summary += f"⏰ **{timestamp}**\n"
            summary += f"📹 Video: {video}\n"
            summary += f"🎭 Behavior: {behavior}\n"
            summary += f"⚡ Level: {level}\n"
            summary += f"💬 {alert.get('message', 'No message')}\n\n"
        
        return summary
    
    def get_alerts_by_severity(self, severity: str) -> str:
        """Get alerts filtered by severity."""
        if not self.alert_data:
            return "No alerts found."
        
        filtered_alerts = [alert for alert in self.alert_data if alert.get('level') == severity]
        
        if not filtered_alerts:
            return f"No {severity} severity alerts found."
        
        summary = f"🚨 **{severity.upper()} Severity Alerts**\n\n"
        for alert in filtered_alerts[-5:]:  # Last 5
            timestamp = alert.get('timestamp', 'Unknown')
            behavior = alert.get('behavior', 'Unknown')
            video = alert.get('video', 'Unknown')
            
            summary += f"⏰ {timestamp} - {video}: {behavior}\n"
        
        return summary
    
    def get_alert_summary(self) -> str:
        """Get overall alert summary."""
        if not self.alert_data:
            return "No alerts found."
        
        total_alerts = len(self.alert_data)
        severity_counts = {}
        
        for alert in self.alert_data:
            level = alert.get('level', 'unknown')
            severity_counts[level] = severity_counts.get(level, 0) + 1
        
        summary = f"🚨 **Alert Summary**\n\n"
        summary += f"📊 Total alerts: {total_alerts}\n\n"
        summary += "📈 **By Severity:**\n"
        for level, count in severity_counts.items():
            summary += f"• {level}: {count}\n"
        
        return summary
    
    def get_processed_videos_summary(self) -> str:
        """Get summary of processed videos."""
        if not self.behavior_data:
            return "No videos have been processed yet."
        
        summary = "📹 **Processed Videos**\n\n"
        summary += f"📊 Total processed: {len(self.behavior_data)}\n\n"
        
        for video_name in self.behavior_data.keys():
            pose_data = self.behavior_data[video_name]
            frames_with_pose = sum(1 for data in pose_data if data['pose_detected'])
            detection_rate = frames_with_pose / len(pose_data) if pose_data else 0
            
            summary += f"• {video_name}: {len(pose_data)} frames, {detection_rate:.1%} pose detection rate\n"
        
        return summary
    
    def get_video_list(self) -> str:
        """Get list of available videos."""
        if not self.video_metadata:
            return "No video metadata available."
        
        summary = "📹 **Available Videos**\n\n"
        for video_name, metadata in self.video_metadata.items():
            size_mb = metadata['size'] / (1024 * 1024)
            modified = metadata['modified'].strftime("%Y-%m-%d %H:%M")
            
            summary += f"• {video_name}\n"
            summary += f"  📁 Size: {size_mb:.1f} MB\n"
            summary += f"  📅 Modified: {modified}\n\n"
        
        return summary
    
    def get_behavior_specific_summary(self, behavior: str) -> str:
        """Get summary for a specific behavior type using the ML classifier."""
        if not self.behavior_data:
            return self.get_random_response('no_data')

        total_occurrences = 0
        videos_with_behavior = []

        for video_name in self.behavior_data:
            behaviors = self._classify_pose_data(video_name)
            if behavior in behaviors:
                count = behaviors[behavior]
                total_occurrences += count
                videos_with_behavior.append((video_name, count))
        
        if total_occurrences == 0:
            return f"No {behavior} behavior detected in any videos."
        
        summary = f"🎭 **{behavior.title()} Behavior Summary**\n\n"
        summary += f"📊 Total occurrences: {total_occurrences}\n"
        summary += f"📹 Videos with {behavior}: {len(videos_with_behavior)}\n\n"
        
        summary += "📈 **By Video:**\n"
        for video_name, count in sorted(videos_with_behavior, key=lambda x: x[1], reverse=True):
            summary += f"• {video_name}: {count} frames\n"
        
        return summary
    
    def get_temporal_behavior_summary(self, time_info: Dict) -> str:
        """Get temporal behavior summary using the ML classifier."""
        if not self.behavior_data:
            return self.get_random_response('no_data')

        out = "⏰ **Temporal Behavior Analysis**\n\n"
        out += "📊 **Behavior Timeline (ML classifier):**\n"

        for video_name, pose_data in self.behavior_data.items():
            behaviors = self._classify_pose_data(video_name)
            out += f"\n📹 **{video_name}:**\n"
            if behaviors:
                for behavior, count in sorted(behaviors.items(), key=lambda x: x[1], reverse=True):
                    pct = count / len(pose_data) if pose_data else 0
                    out += f"• {behavior}: {pct:.1%} of video duration\n"
            else:
                out += "• No behaviors classified (run model training first).\n"

        return out
    
    def get_statistical_summary(self) -> str:
        """Get statistical summary of behavior data."""
        if not self.behavior_data:
            return self.get_random_response('no_data')
        
        summary = "📊 **Statistical Summary**\n\n"
        
        # Calculate statistics
        total_frames = sum(len(pose_data) for pose_data in self.behavior_data.values())
        total_poses = sum(
            sum(1 for data in pose_data if data['pose_detected'])
            for pose_data in self.behavior_data.values()
        )
        
        summary += f"📈 **Overall Statistics:**\n"
        summary += f"• Total frames analyzed: {total_frames:,}\n"
        summary += f"• Total poses detected: {total_poses:,}\n"
        summary += f"• Average pose detection rate: {total_poses/total_frames:.1%}\n"
        summary += f"• Videos analyzed: {len(self.behavior_data)}\n"
        summary += f"• Average frames per video: {total_frames/len(self.behavior_data):.0f}\n"
        
        return summary
    
    def get_random_response(self, response_type: str) -> str:
        """Get a random response from the specified type."""
        responses = self.response_templates.get(response_type, ["I'm not sure how to respond to that."])
        return np.random.choice(responses)

def main():
    """Main function to demonstrate chatbot."""
    chatbot = BehaviorChatbot()
    
    print("🤖 Behavior Detection Chatbot")
    print("="*50)
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Goodbye! Have a great day!")
                break
            
            if not query:
                continue
            
            response = chatbot.process_query(query)
            print(f"🤖 {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n🤖 Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"🤖 Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    main() 