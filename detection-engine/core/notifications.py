import logging
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

LOGGER = logging.getLogger(__name__)


class NotificationManager:
    """Manages notifications for fire detection events."""

    def __init__(self, sns_topic_arn=None):
        # Use provided ARN or fall back to environment variable
        self.sns_topic_arn = sns_topic_arn or os.getenv("SNS_TOPIC_ARN")
        self.sns_client = None
        if self.sns_topic_arn:
            self.initialize_sns(self.sns_topic_arn)
        else:
            LOGGER.warning(
                "SNS_TOPIC_ARN not provided or found in environment. SNS notifications disabled."
            )

    def initialize_sns(self, sns_topic_arn):
        """Initialize AWS SNS client."""
        try:
            # boto3 will automatically check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
            self.sns_client = boto3.client(
                "sns", region_name=os.getenv("AWS_REGION", "us-east-1")
            )
            # Verify topic exists
            self.sns_client.get_topic_attributes(TopicArn=sns_topic_arn)
            LOGGER.info(f"AWS SNS initialized with topic: {sns_topic_arn}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize AWS SNS: {e}")
            self.sns_client = None

    def send_sns_notification(self, severity, timestamp):
        """Send notification via AWS SNS."""
        if self.sns_client is None:
            LOGGER.warning("SNS client not initialized, skipping notification")
            return

        try:
            message = (
                f"🔥 FIRE SEVERITY ALERT: {severity}\n"
                f"Timestamp: {timestamp}\n"
                f"System: PyroGuardian Edge-AI\n"
                f"Action Required: Please check the live stream immediately."
            )
            self.sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Message=message,
                Subject=f"PyroGuardian: {severity} Severity Alert",
            )
            LOGGER.info(f"Sent SNS notification: {severity} at {timestamp}")
        except Exception as e:
            LOGGER.error(f"Failed to send SNS notification: {e}")


def determine_severity(detected_classes):
    """
    Determine severity based on detected classes.
    Logic:
    - High: Fire + Person + Other
    - Medium: (Fire + Person) or (Fire + Other)
    - Low: Just Fire
    """
    has_fire = "fire" in detected_classes
    has_person = "person" in detected_classes
    has_other = any(cls not in ["fire", "person"] for cls in detected_classes)

    if has_fire and has_person and has_other:
        return "High"
    elif (has_fire and has_person) or (has_fire and has_other and not has_person):
        return "Medium"
    elif has_fire:
        return "Low"
    return None
