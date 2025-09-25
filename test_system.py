#!/usr/bin/env python3
"""
Test script for Human Behavior Detection System
===============================================

This script tests all major components of the system to ensure they're working correctly.
"""

import sys
import os
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from preprocessing.extract_frames import FrameExtractor
        print("✅ FrameExtractor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FrameExtractor: {e}")
        return False
    
    try:
        from preprocessing.extract_pose import PoseExtractor
        print("✅ PoseExtractor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PoseExtractor: {e}")
        return False
    
    try:
        from ai_model.predict_behavior import BehaviorClassifier
        print("✅ BehaviorClassifier imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import BehaviorClassifier: {e}")
        return False
    
    try:
        from alerts.alert_trigger import AlertTrigger, BehaviorMonitor
        print("✅ AlertTrigger and BehaviorMonitor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AlertTrigger: {e}")
        return False
    
    try:
        from chatbot.chatbot_interface import BehaviorChatbot
        print("✅ BehaviorChatbot imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import BehaviorChatbot: {e}")
        return False
    
    return True

def test_directories():
    """Test that required directories exist and can be created."""
    print("\n📁 Testing directories...")
    
    directories = [
        "data/raw_videos",
        "data/extracted_frames",
        "data/keypoints",
        "data/labels",
        "data/logs",
        "preprocessing",
        "ai_model",
        "alerts",
        "chatbot"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory {directory} ready")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def test_video_files():
    """Test that video files are present."""
    print("\n📹 Testing video files...")
    
    raw_videos_dir = Path("data/raw_videos")
    video_files = list(raw_videos_dir.glob("*.mp4"))
    
    if not video_files:
        print("⚠️  No MP4 video files found in data/raw_videos/")
        print("   This is expected if you haven't added videos yet")
        return True
    else:
        print(f"✅ Found {len(video_files)} video files:")
        for video_file in video_files[:5]:  # Show first 5
            print(f"   - {video_file.name}")
        if len(video_files) > 5:
            print(f"   ... and {len(video_files) - 5} more")
        return True

def test_frame_extractor():
    """Test frame extraction functionality."""
    print("\n🖼️  Testing frame extractor...")
    
    try:
        from preprocessing.extract_frames import FrameExtractor
        
        extractor = FrameExtractor()
        print("✅ FrameExtractor initialized successfully")
        
        # Test with a small sample if videos exist
        raw_videos_dir = Path("data/raw_videos")
        video_files = list(raw_videos_dir.glob("*.mp4"))
        
        if video_files:
            print(f"   Found {len(video_files)} videos to test with")
            return True
        else:
            print("   No videos available for testing")
            return True
            
    except Exception as e:
        print(f"❌ Frame extractor test failed: {e}")
        return False

def test_pose_extractor():
    """Test pose extraction functionality."""
    print("\n🎭 Testing pose extractor...")
    
    try:
        from preprocessing.extract_pose import PoseExtractor
        
        extractor = PoseExtractor()
        print("✅ PoseExtractor initialized successfully")
        
        # Check if we have extracted frames to test with
        frames_dir = Path("data/extracted_frames")
        if frames_dir.exists() and any(frames_dir.iterdir()):
            print("   Found extracted frames for testing")
        else:
            print("   No extracted frames available (run frame extraction first)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pose extractor test failed: {e}")
        return False

def test_behavior_classifier():
    """Test behavior classification functionality."""
    print("\n🤖 Testing behavior classifier...")
    
    try:
        from ai_model.predict_behavior import BehaviorClassifier, create_synthetic_training_data
        
        classifier = BehaviorClassifier()
        print("✅ BehaviorClassifier initialized successfully")
        
        # Test synthetic data creation
        synthetic_data = create_synthetic_training_data()
        print(f"✅ Synthetic data created for {len(synthetic_data)} behaviors")
        
        return True
        
    except Exception as e:
        print(f"❌ Behavior classifier test failed: {e}")
        return False

def test_alert_system():
    """Test alert system functionality."""
    print("\n🚨 Testing alert system...")
    
    try:
        from alerts.alert_trigger import AlertTrigger, BehaviorMonitor
        
        alert_trigger = AlertTrigger()
        print("✅ AlertTrigger initialized successfully")
        
        monitor = BehaviorMonitor(alert_trigger)
        print("✅ BehaviorMonitor initialized successfully")
        
        # Test alert configuration
        config = alert_trigger.config
        print(f"✅ Alert configuration loaded: {len(config)} settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert system test failed: {e}")
        return False

def test_chatbot():
    """Test chatbot functionality."""
    print("\n💬 Testing chatbot...")
    
    try:
        from chatbot.chatbot_interface import BehaviorChatbot
        
        chatbot = BehaviorChatbot()
        print("✅ BehaviorChatbot initialized successfully")
        
        # Test basic query processing
        test_query = "Hello"
        response = chatbot.process_query(test_query)
        print(f"✅ Chatbot processed test query: '{test_query}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'cv2',
        'mediapipe',
        'torch',
        'numpy',
        'sklearn',
        'tqdm',
        'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install them with: pip install -r requirements.txt")
        return False
    
    return True

def run_full_test():
    """Run all tests."""
    print("🧪 HUMAN BEHAVIOR DETECTION SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Directories", test_directories),
        ("Video Files", test_video_files),
        ("Frame Extractor", test_frame_extractor),
        ("Pose Extractor", test_pose_extractor),
        ("Behavior Classifier", test_behavior_classifier),
        ("Alert System", test_alert_system),
        ("Chatbot", test_chatbot)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Add MP4 videos to data/raw_videos/")
        print("   2. Run: python main.py --mode full_pipeline")
        print("   3. Start chatbot: python main.py --mode chat")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Check file permissions")
        print("   3. Verify Python version (3.7+)")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1) 