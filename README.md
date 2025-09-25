# Human Behavior Detection and Alert System

A comprehensive AI-powered system for detecting and analyzing human behaviors in video footage using computer vision and machine learning techniques.

## 🎯 Project Overview

This system automatically analyzes video files to:
- Extract video frames at configurable intervals
- Detect human poses using MediaPipe
- Classify behaviors (normal, falling, fighting, loitering, running, walking)
- Trigger alerts for suspicious behaviors
- Provide an interactive chatbot for querying analysis results

## 🏗️ System Architecture

```
human_behavior_detection_project/
├── data/
│   ├── raw_videos/           # Input MP4 video files
│   ├── extracted_frames/     # Extracted video frames
│   ├── keypoints/           # Pose keypoint data
│   ├── labels/              # Behavior labels (for training)
│   └── logs/                # System logs and alerts
├── preprocessing/
│   ├── extract_frames.py    # Video frame extraction
│   └── extract_pose.py      # Pose keypoint extraction
├── ai_model/
│   └── predict_behavior.py  # Behavior classification model
├── alerts/
│   └── alert_trigger.py     # Alert system
├── chatbot/
│   └── chatbot_interface.py # Interactive chatbot
├── main.py                  # Main orchestration script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd human_behavior_detection_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Videos

Place your MP4 video files in the `data/raw_videos/` directory.

### 3. Start Interactive Mode (Recommended)

```bash
# Start the interactive menu system
python main.py --mode interactive
```

This will show you a menu with options:
- 📹 Analyze Saved Videos
- 🎥 Real-time Camera Analysis  
- 🔄 Run Full Pipeline
- 💬 Chat with AI Assistant
- 📊 System Status
- 🧪 Test System

### 4. Alternative: Run Individual Components

```bash
# Run complete analysis pipeline
python main.py --mode full_pipeline

# Start interactive chatbot
python main.py --mode chat

# Check system status
python main.py --status
```

## 📋 Usage Modes

### Interactive Mode (Recommended)
```bash
python main.py --mode interactive
```
Start the interactive menu system with all features:
- 📹 Analyze saved videos
- 🎥 Real-time camera analysis
- 🔄 Run full pipeline
- 💬 Chat with AI assistant
- 📊 System status
- 🧪 Test system

### Saved Video Analysis
```bash
python main.py --mode extract_frames --fps 1
python main.py --mode extract_pose
python main.py --mode analyze
```
Or use the interactive menu to select and analyze specific videos.

### Real-time Camera Analysis
```bash
python main.py --mode interactive
# Then select option 2
```
Use your webcam for live behavior detection with pose visualization.

### Full Pipeline
```bash
python main.py --mode full_pipeline
```
Runs the complete workflow: frame extraction → pose detection → behavior classification → alert monitoring.

### Chatbot Interface
```bash
python main.py --mode chat
```
Start an interactive chatbot to query analysis results.

### System Status
```bash
python main.py --status
```
Check the current status of all system components.

## 🤖 Chatbot Queries

The chatbot can answer questions like:

- **"What behaviors were detected in the videos?"**
- **"Show me recent alerts"**
- **"How many videos have been processed?"**
- **"What happened in video X?"**
- **"Show me statistics about falling behavior"**
- **"What are the most common behaviors detected?"**

## 🚨 Alert System

The system automatically triggers alerts for suspicious behaviors:

| Behavior | Severity | Threshold |
|----------|----------|-----------|
| Falling | High | 30% |
| Fighting | Critical | 20% |
| Loitering | Medium | 50% |
| Running | Medium | 40% |

Alerts include:
- Console notifications with color coding
- Log file entries
- Email notifications (configurable)
- Cooldown periods to prevent spam

## 🔧 Configuration

### Alert Configuration
Edit `alerts/alert_config.json` to customize:
- Email notification settings
- Alert thresholds
- Cooldown periods
- Log file locations

### Frame Extraction
Control frame extraction rate:
```bash
python main.py --mode full_pipeline --fps 2  # 2 frames per second
```

### Model Training
Choose between synthetic and real data:
```bash
# Use synthetic data (default)
python main.py --mode train_model

# Use real data (requires labeled data)
python main.py --mode train_model --no-synthetic
```

## 📊 Output Files

### Frame Extraction
- **Location**: `data/extracted_frames/<video_name>/`
- **Format**: JPEG images named `frame_XXXXXX.jpg`

### Pose Data
- **Location**: `data/keypoints/<video_name>/`
- **Files**: 
  - `pose_data.json` - Human-readable pose data
  - `pose_data.pkl` - Binary format for fast loading

### Behavior Analysis
- **Model**: `ai_model/behavior_classifier.pkl`
- **Logs**: `data/logs/main.log`
- **Alerts**: `data/logs/alerts.log`

## 🛠️ Technical Details

### Pose Detection
- **Library**: MediaPipe Pose
- **Landmarks**: 33 body keypoints
- **Features**: x, y, z coordinates + visibility scores

### Behavior Classification
- **Algorithm**: Random Forest Classifier
- **Features**: Pose landmark coordinates
- **Classes**: normal, falling, fighting, loitering, running, walking

### Alert System
- **Types**: Console, Email, Log file
- **Severity Levels**: Low, Medium, High, Critical
- **Cooldown**: Configurable time periods

## 📈 Performance

### Processing Speed
- **Frame Extraction**: ~100-500 fps (depending on video resolution)
- **Pose Detection**: ~10-30 fps per frame
- **Behavior Classification**: ~1000+ predictions/second

### Accuracy
- **Pose Detection**: >90% for clear human figures
- **Behavior Classification**: Varies by behavior type (requires training data)

## 🔍 Troubleshooting

### Common Issues

1. **No videos found**
   - Ensure MP4 files are in `data/raw_videos/`
   - Check file permissions

2. **Pose detection fails**
   - Verify video quality and lighting
   - Check MediaPipe installation

3. **Model training errors**
   - Use synthetic data for testing: `--no-synthetic` flag
   - Check available memory

4. **Alert system not working**
   - Verify alert configuration file
   - Check log files for errors

### Log Files
- **Main logs**: `data/logs/main.log`
- **Alert logs**: `data/logs/alerts.log`
- **Chatbot logs**: Console output

## 🧪 Testing

### Synthetic Data
The system includes synthetic training data for testing:
```bash
python main.py --mode train_model  # Uses synthetic data by default
```

### Real Data
For production use, prepare labeled training data:
1. Extract frames from videos
2. Extract pose keypoints
3. Label behaviors manually
4. Train model with real data

## 🔮 Future Enhancements

- [ ] Real-time video processing
- [ ] Multi-person detection
- [ ] Advanced behavior patterns
- [ ] Web interface
- [ ] Mobile app integration
- [ ] Cloud deployment
- [ ] Custom behavior training

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and support:
- Check the troubleshooting section
- Review log files for errors
- Open an issue on GitHub

---

**Note**: This system is designed for research and educational purposes. For production deployment, ensure compliance with privacy laws and regulations. 