import pytest
from unittest.mock import MagicMock, patch
from core.notifications import NotificationManager, determine_severity

class TestNotificationManager:
    @pytest.fixture
    def mock_boto3(self):
        with patch("boto3.client") as mock:
            yield mock

    def test_init_with_env_var(self, mock_boto3, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:MyTopic")
        manager = NotificationManager()
        
        assert manager.sns_topic_arn == "arn:aws:sns:us-east-1:123456789012:MyTopic"
        mock_boto3.assert_called_once_with("sns", region_name="us-east-1")

    def test_init_without_arn(self, mock_boto3, monkeypatch):
        """Test initialization without ARN logs warning."""
        monkeypatch.delenv("SNS_TOPIC_ARN", raising=False)
        manager = NotificationManager()
        
        assert manager.sns_topic_arn is None
        # Should not initialize client if ARN is missing
        assert manager.sns_client is None

    def test_send_notification_success(self, mock_boto3):
        """Test successful notification sending."""
        arn = "arn:aws:sns:test"
        manager = NotificationManager(sns_topic_arn=arn)
        
        # Mock the sns client instance returned by boto3.client
        mock_sns_instance = mock_boto3.return_value
        manager.sns_client = mock_sns_instance

        manager.send_sns_notification("High", "2023-01-01 12:00:00")

        mock_sns_instance.publish.assert_called_once()
        call_args = mock_sns_instance.publish.call_args[1]
        assert call_args["TopicArn"] == arn
        assert "High" in call_args["Message"]
        assert "PyroGuardian" in call_args["Subject"]

def test_determine_severity():
    """Test severity logic."""
    # High Severity
    assert determine_severity({"fire", "person", "other"}) == "High"
    
    # Medium Severity
    assert determine_severity({"fire", "person"}) == "Medium"
    assert determine_severity({"fire", "other"}) == "Medium" # Fire + Other
    
    # Low Severity
    assert determine_severity({"fire"}) == "Low"
    
    # No Fire
    assert determine_severity({"person"}) is None
    assert determine_severity(set()) is None
