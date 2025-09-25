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

class AlertTrigger:
    """Alert system for suspicious behavior detection."""
    
    def __init__(self, config_file="alerts/alert_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.alert_history = []
        self.suspicious_behaviors = {
            'falling': {'severity': 'high', 'threshold': 0.3},
            'fighting': {'severity': 'critical', 'threshold': 0.2},
            'loitering': {'severity': 'medium', 'threshold': 0.5},
            'running': {'severity': 'medium', 'threshold': 0.4}
        }
        
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
            'notification_cooldown': 300,  # 5 minutes
            'alert_levels': {
                'low': {'color': 'green', 'priority': 1},
                'medium': {'color': 'yellow', 'priority': 2},
                'high': {'color': 'orange', 'priority': 3},
                'critical': {'color': 'red', 'priority': 4}
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config file
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def check_behavior_alerts(self, behavior_summary: Dict, video_name: str = "Unknown") -> List[Dict]:
        """
        Check if behaviors in the summary trigger alerts.
        
        Args:
            behavior_summary: Summary of detected behaviors
            video_name: Name of the video being analyzed
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        if not behavior_summary or 'behavior_percentages' not in behavior_summary:
            return alerts
        
        for behavior, percentage in behavior_summary['behavior_percentages'].items():
            if behavior in self.suspicious_behaviors:
                threshold = self.suspicious_behaviors[behavior]['threshold']
                severity = self.suspicious_behaviors[behavior]['severity']
                
                if percentage >= threshold:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'video_name': video_name,
                        'behavior': behavior,
                        'percentage': percentage,
                        'severity': severity,
                        'threshold': threshold,
                        'message': f"Suspicious behavior detected: {behavior} ({percentage:.1%})"
                    }
                    alerts.append(alert)
        
        return alerts

    def compute_violence_level(self, behavior_summary: Dict) -> float:
        """Compute a simple violence level score from behavior percentages.

        Heuristic: combine fighting and falling as strong signals, running medium.
        Returns 0..1.
        """
        if not behavior_summary or 'behavior_percentages' not in behavior_summary:
            return 0.0
        bp = behavior_summary['behavior_percentages']
        score = 0.0
        score += 1.0 * bp.get('fighting', 0.0)
        score += 0.8 * bp.get('falling', 0.0)
        score += 0.3 * bp.get('running', 0.0)
        return max(0.0, min(1.0, score))
    
    def trigger_alert(self, alert: Dict) -> bool:
        """
        Trigger an alert with the specified action.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if alert was successfully triggered
        """
        try:
            # Check cooldown
            if self.is_in_cooldown(alert):
                logger.info(f"Alert in cooldown: {alert['behavior']}")
                return False
            
            # Log alert
            if self.config['log_alerts']:
                self.log_alert(alert)
            
            # Send email notification
            if self.config['email_enabled']:
                self.send_email_alert(alert)
            
            # Print alert
            self.print_alert(alert)
            
            # Add to history
            self.alert_history.append(alert)
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
            return False
    
    def is_in_cooldown(self, alert: Dict) -> bool:
        """Check if alert is in cooldown period."""
        cooldown_seconds = self.config['notification_cooldown']
        current_time = time.time()
        
        # Check recent alerts for the same behavior
        for recent_alert in self.alert_history[-10:]:  # Check last 10 alerts
            if (recent_alert['behavior'] == alert['behavior'] and 
                recent_alert['video_name'] == alert['video_name']):
                alert_time = datetime.fromisoformat(recent_alert['timestamp']).timestamp()
                if current_time - alert_time < cooldown_seconds:
                    return True
        
        return False
    
    def log_alert(self, alert: Dict):
        """Log alert to file."""
        log_file = Path(self.config['alert_log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            'timestamp': alert['timestamp'],
            'level': alert['severity'],
            'video': alert['video_name'],
            'behavior': alert['behavior'],
            'percentage': alert['percentage'],
            'message': alert['message']
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def send_email_alert(self, alert: Dict):
        """Send email alert."""
        if not self.config['email_enabled']:
            return
        
        email_config = self.config['email_settings']
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = ', '.join(email_config['recipient_emails'])
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
        
        # Send email
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
        """Print alert to console."""
        severity_colors = {
            'low': '\033[92m',      # Green
            'medium': '\033[93m',   # Yellow
            'high': '\033[91m',     # Red
            'critical': '\033[95m'  # Magenta
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
        """
        Get summary of alerts in the specified time window.
        
        Args:
            time_window_hours: Hours to look back
            
        Returns:
            Alert summary
        """
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (time_window_hours * 3600)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']).timestamp() > cutoff_time
        ]
        
        # Group by behavior
        behavior_counts = {}
        severity_counts = {}
        
        for alert in recent_alerts:
            behavior = alert['behavior']
            severity = alert['severity']
            
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'time_window_hours': time_window_hours,
            'behavior_counts': behavior_counts,
            'severity_counts': severity_counts,
            'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
        }
    
    def clear_alert_history(self):
        """Clear alert history."""
        self.alert_history = []
        logger.info("Alert history cleared")

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