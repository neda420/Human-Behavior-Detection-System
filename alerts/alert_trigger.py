import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default thresholds used when config file is absent or lacks the section.
_DEFAULT_THRESHOLDS = {
    'falling':   {'severity': 'high',     'threshold': 0.30},
    'fighting':  {'severity': 'critical', 'threshold': 0.20},
    'loitering': {'severity': 'medium',   'threshold': 0.50},
    'running':   {'severity': 'medium',   'threshold': 0.40},
}


class AlertTrigger:
    """Alert system for suspicious behavior detection."""

    def __init__(self, config_file="alerts/alert_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.alert_history: List[Dict] = []
        # Per-class, per-source cooldown tracker: (behavior, video_name) → last alert unix-time
        self._cooldown_tracker: Dict[tuple, float] = {}
        # Load behavior thresholds from config (fall back to defaults if absent)
        self.suspicious_behaviors = self._load_suspicious_behaviors()

    def _load_suspicious_behaviors(self) -> Dict[str, Dict]:
        """Load per-behavior thresholds from config, falling back to defaults."""
        cfg_thresholds = self.config.get('behavior_thresholds', {})
        if cfg_thresholds:
            logger.info(f"Loaded {len(cfg_thresholds)} behavior thresholds from config.")
            return cfg_thresholds
        logger.info("No behavior_thresholds in config; using built-in defaults.")
        return dict(_DEFAULT_THRESHOLDS)

    def load_config(self):
        """Load alert configuration."""
        default_config = {
            'email_enabled': False,
            'email_settings': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'recipient_emails': []
            },
            'log_alerts': True,
            'alert_log_file': 'data/logs/alerts.log',
            'notification_cooldown': 300,
            'alert_levels': {
                'low':      {'color': 'green',   'priority': 1},
                'medium':   {'color': 'yellow',  'priority': 2},
                'high':     {'color': 'orange',  'priority': 3},
                'critical': {'color': 'red',     'priority': 4},
            },
            'behavior_thresholds': _DEFAULT_THRESHOLDS,
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def check_behavior_alerts(self, behavior_summary: Dict, video_name: str = "Unknown") -> List[Dict]:
        """Check if behaviors in the summary trigger alerts."""
        alerts = []

        if not behavior_summary or 'behavior_percentages' not in behavior_summary:
            return alerts

        for behavior, percentage in behavior_summary['behavior_percentages'].items():
            if behavior in self.suspicious_behaviors:
                threshold = self.suspicious_behaviors[behavior]['threshold']
                severity  = self.suspicious_behaviors[behavior]['severity']

                if percentage >= threshold:
                    alert = {
                        'timestamp':   datetime.now().isoformat(),
                        'video_name':  video_name,
                        'behavior':    behavior,
                        'percentage':  percentage,
                        'severity':    severity,
                        'threshold':   threshold,
                        'message':     f"Suspicious behavior detected: {behavior} ({percentage:.1%})",
                    }
                    alerts.append(alert)

        return alerts

    def compute_violence_level(self, behavior_summary: Dict) -> float:
        """Compute a violence level score (0–1) from calibrated behavior probabilities."""
        if not behavior_summary or 'behavior_percentages' not in behavior_summary:
            return 0.0
        bp = behavior_summary['behavior_percentages']
        score = (
            1.0 * bp.get('fighting', 0.0)
            + 0.8 * bp.get('falling', 0.0)
            + 0.3 * bp.get('running', 0.0)
        )
        return max(0.0, min(1.0, score))

    def trigger_alert(self, alert: Dict) -> bool:
        """Trigger an alert (logs, prints, optionally emails)."""
        try:
            if self.is_in_cooldown(alert):
                logger.info(f"Alert in cooldown: {alert['behavior']} / {alert['video_name']}")
                return False

            if self.config['log_alerts']:
                self.log_alert(alert)

            if self.config['email_enabled']:
                self.send_email_alert(alert)

            self.print_alert(alert)
            self.alert_history.append(alert)

            # Update per-class, per-source cooldown tracker
            key = (alert['behavior'], alert['video_name'])
            self._cooldown_tracker[key] = time.time()

            return True

        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
            return False

    def is_in_cooldown(self, alert: Dict) -> bool:
        """Check if a (behavior, video_name) pair is still within its cooldown window."""
        cooldown_seconds = self.config['notification_cooldown']
        key = (alert['behavior'], alert['video_name'])
        last_time = self._cooldown_tracker.get(key)
        if last_time is None:
            return False
        return (time.time() - last_time) < cooldown_seconds

    def log_alert(self, alert: Dict):
        """Log alert to file."""
        log_file = Path(self.config['alert_log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            'timestamp': alert['timestamp'],
            'level':     alert['severity'],
            'video':     alert['video_name'],
            'behavior':  alert['behavior'],
            'percentage': alert['percentage'],
            'message':   alert['message'],
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def send_email_alert(self, alert: Dict):
        """Send email alert."""
        if not self.config['email_enabled']:
            return

        email_config = self.config['email_settings']

        msg = MIMEMultipart()
        msg['From']    = email_config['sender_email']
        msg['To']      = ', '.join(email_config['recipient_emails'])
        msg['Subject'] = f"ALERT: {alert['severity'].upper()} - {alert['behavior']} detected"

        body = f"""
        SUSPICIOUS BEHAVIOR ALERT

        Video: {alert['video_name']}
        Behavior: {alert['behavior']}
        Severity: {alert['severity']}
        Percentage: {alert['percentage']:.1%}
        Threshold: {alert['threshold']:.1%}
        Time: {alert['timestamp']}

        Message: {alert['message']}
        """

        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            logger.info(f"Email alert sent for {alert['behavior']}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def print_alert(self, alert: Dict):
        """Print alert to console with colour."""
        severity_colors = {
            'low':      '\033[92m',
            'medium':   '\033[93m',
            'high':     '\033[91m',
            'critical': '\033[95m',
        }

        color = severity_colors.get(alert['severity'], '\033[0m')
        reset = '\033[0m'

        print(f"\n{color}{'='*60}")
        print(f"🚨 ALERT: {alert['severity'].upper()} 🚨")
        print(f"{'='*60}{reset}")
        print(f"📹 Video: {alert['video_name']}")
        print(f"🎭 Behavior: {alert['behavior']}")
        print(f"📊 Percentage: {alert['percentage']:.1%}")
        print(f"⚡ Severity: {alert['severity']}")
        print(f"⏰ Time: {alert['timestamp']}")
        print(f"💬 Message: {alert['message']}")
        print(f"{color}{'='*60}{reset}\n")

    def get_alert_summary(self, time_window_hours: int = 24) -> Dict:
        """Get summary of alerts in the specified time window."""
        current_time   = datetime.now()
        cutoff_time    = current_time.timestamp() - (time_window_hours * 3600)

        recent_alerts = [
            a for a in self.alert_history
            if datetime.fromisoformat(a['timestamp']).timestamp() > cutoff_time
        ]

        behavior_counts  = {}
        severity_counts  = {}

        for alert in recent_alerts:
            behavior_counts[alert['behavior']] = behavior_counts.get(alert['behavior'], 0) + 1
            severity_counts[alert['severity']] = severity_counts.get(alert['severity'], 0) + 1

        return {
            'total_alerts':     len(recent_alerts),
            'time_window_hours': time_window_hours,
            'behavior_counts':  behavior_counts,
            'severity_counts':  severity_counts,
            'recent_alerts':    recent_alerts[-10:],
        }

    def clear_alert_history(self):
        """Clear in-memory alert history and cooldown tracker."""
        self.alert_history.clear()
        self._cooldown_tracker.clear()
        logger.info("Alert history and cooldown tracker cleared.")

class BehaviorMonitor:
    """Monitor behavior detection results and trigger alerts."""
    
    def __init__(self, alert_trigger: AlertTrigger):
        self.alert_trigger = alert_trigger
        self.monitoring_active = False
    
    def start_monitoring(self):
        """Start behavior monitoring."""
        self.monitoring_active = True
        logger.info("Behavior monitoring started")
    
    def stop_monitoring(self):
        """Stop behavior monitoring."""
        self.monitoring_active = False
        logger.info("Behavior monitoring stopped")
    
    def process_behavior_results(self, behavior_summary: Dict, video_name: str = "Unknown"):
        """
        Process behavior detection results and trigger alerts if needed.
        
        Args:
            behavior_summary: Summary of detected behaviors
            video_name: Name of the video being analyzed
        """
        if not self.monitoring_active:
            return
        
        # Check for alerts
        alerts = self.alert_trigger.check_behavior_alerts(behavior_summary, video_name)
        
        # Trigger alerts
        for alert in alerts:
            self.alert_trigger.trigger_alert(alert)
        
        return alerts
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            'monitoring_active': self.monitoring_active,
            'alert_summary': self.alert_trigger.get_alert_summary()
        }

def main():
    """Main function to demonstrate alert system."""
    # Initialize alert trigger
    alert_trigger = AlertTrigger()
    
    # Initialize behavior monitor
    monitor = BehaviorMonitor(alert_trigger)
    monitor.start_monitoring()
    
    # Simulate behavior detection results
    test_behaviors = [
        {
            'behavior_percentages': {
                'normal': 0.6,
                'falling': 0.35,  # Should trigger alert
                'walking': 0.05
            }
        },
        {
            'behavior_percentages': {
                'normal': 0.3,
                'fighting': 0.25,  # Should trigger alert
                'running': 0.45    # Should trigger alert
            }
        }
    ]
    
    video_names = ["test_video_1.mp4", "test_video_2.mp4"]
    
    print("Testing Alert System...")
    print("="*50)
    
    for i, (behavior_summary, video_name) in enumerate(zip(test_behaviors, video_names)):
        print(f"\nProcessing {video_name}...")
        alerts = monitor.process_behavior_results(behavior_summary, video_name)
        
        if alerts:
            print(f"Triggered {len(alerts)} alerts")
        else:
            print("No alerts triggered")
    
    # Get monitoring status
    status = monitor.get_monitoring_status()
    print(f"\nMonitoring Status: {'Active' if status['monitoring_active'] else 'Inactive'}")
    print(f"Total alerts in last 24h: {status['alert_summary']['total_alerts']}")
    
    monitor.stop_monitoring()

if __name__ == "__main__":
    main() 