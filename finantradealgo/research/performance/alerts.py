"""
Performance Alert System.

Manages alert subscriptions, notifications, and delivery for performance issues.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from finantradealgo.research.performance.models import PerformanceAlert


class AlertChannel(str, Enum):
    """Alert delivery channels."""

    CONSOLE = "console"  # Print to console
    FILE = "file"  # Write to file
    EMAIL = "email"  # Send email (requires SMTP config)
    WEBHOOK = "webhook"  # POST to webhook URL
    SLACK = "slack"  # Send to Slack channel


class AlertPriority(str, Enum):
    """Alert priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSubscription:
    """
    Alert subscription configuration.

    Defines when and how alerts should be delivered.
    """

    def __init__(
        self,
        strategy_id: str,
        channels: List[AlertChannel],
        severity_filter: Optional[List[str]] = None,
        alert_type_filter: Optional[List[str]] = None,
        min_interval_minutes: int = 15,
        enabled: bool = True,
    ):
        """
        Initialize alert subscription.

        Args:
            strategy_id: Strategy to monitor
            channels: Delivery channels for alerts
            severity_filter: Only alert on these severities (None = all)
            alert_type_filter: Only alert on these types (None = all)
            min_interval_minutes: Minimum interval between same alerts
            enabled: Whether subscription is active
        """
        self.strategy_id = strategy_id
        self.channels = channels
        self.severity_filter = severity_filter or ["warning", "critical"]
        self.alert_type_filter = alert_type_filter
        self.min_interval_minutes = min_interval_minutes
        self.enabled = enabled

        # Track last alert time for rate limiting
        self.last_alert_times: Dict[str, datetime] = {}

    def should_send(self, alert: PerformanceAlert) -> bool:
        """
        Check if alert should be sent based on subscription filters.

        Args:
            alert: Performance alert

        Returns:
            True if alert should be sent
        """
        if not self.enabled:
            return False

        if alert.strategy_id != self.strategy_id:
            return False

        # Check severity filter
        if self.severity_filter and alert.severity not in self.severity_filter:
            return False

        # Check alert type filter
        if self.alert_type_filter and alert.alert_type not in self.alert_type_filter:
            return False

        # Check rate limiting
        alert_key = f"{alert.alert_type}_{alert.severity}"
        last_time = self.last_alert_times.get(alert_key)

        if last_time:
            elapsed = (datetime.utcnow() - last_time).total_seconds() / 60
            if elapsed < self.min_interval_minutes:
                return False

        # Update last alert time
        self.last_alert_times[alert_key] = datetime.utcnow()

        return True


class AlertHandler:
    """
    Base class for alert delivery handlers.
    """

    def send(self, alert: PerformanceAlert, subscription: AlertSubscription) -> bool:
        """
        Send alert via this handler.

        Args:
            alert: Alert to send
            subscription: Subscription configuration

        Returns:
            True if sent successfully
        """
        raise NotImplementedError


class ConsoleAlertHandler(AlertHandler):
    """Print alerts to console."""

    def send(self, alert: PerformanceAlert, subscription: AlertSubscription) -> bool:
        """Print alert to console."""
        severity_icon = {
            "warning": "⚠️ ",
            "critical": "🚨",
        }.get(alert.severity, "ℹ️ ")

        print(f"\n{severity_icon} PERFORMANCE ALERT [{alert.severity.upper()}]")
        print(f"Strategy: {alert.strategy_id}")
        print(f"Time: {alert.alert_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Type: {alert.alert_type}")
        print(f"Message: {alert.message}")
        if alert.recommended_action:
            print(f"Action: {alert.recommended_action}")
        print("-" * 70)

        return True


class FileAlertHandler(AlertHandler):
    """Write alerts to file."""

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize file handler.

        Args:
            log_dir: Directory for alert logs (default: outputs/alerts)
        """
        if log_dir is None:
            log_dir = Path("outputs") / "alerts"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def send(self, alert: PerformanceAlert, subscription: AlertSubscription) -> bool:
        """Write alert to file."""
        # Create daily log file
        log_file = self.log_dir / f"alerts_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

        alert_data = alert.to_dict()
        alert_data["subscription_channels"] = [c.value for c in subscription.channels]

        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert_data) + "\n")

        return True


class WebhookAlertHandler(AlertHandler):
    """Send alerts to webhook URL."""

    def __init__(self, webhook_url: str):
        """
        Initialize webhook handler.

        Args:
            webhook_url: URL to POST alerts to
        """
        self.webhook_url = webhook_url

    def send(self, alert: PerformanceAlert, subscription: AlertSubscription) -> bool:
        """Send alert to webhook."""
        try:
            import requests

            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )

            return response.status_code == 200

        except Exception as e:
            print(f"[WARN] Failed to send webhook alert: {e}")
            return False


class SlackAlertHandler(AlertHandler):
    """Send alerts to Slack channel."""

    def __init__(self, webhook_url: str):
        """
        Initialize Slack handler.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url

    def send(self, alert: PerformanceAlert, subscription: AlertSubscription) -> bool:
        """Send alert to Slack."""
        try:
            import requests

            # Format Slack message
            severity_emoji = {
                "warning": ":warning:",
                "critical": ":rotating_light:",
            }.get(alert.severity, ":information_source:")

            color = {
                "warning": "#FFA500",  # Orange
                "critical": "#FF0000",  # Red
            }.get(alert.severity, "#0000FF")  # Blue

            message = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{severity_emoji} Performance Alert: {alert.alert_type}",
                        "fields": [
                            {"title": "Strategy", "value": alert.strategy_id, "short": True},
                            {"title": "Severity", "value": alert.severity.upper(), "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Recommended Action", "value": alert.recommended_action or "N/A", "short": False},
                        ],
                        "footer": "Performance Monitoring System",
                        "ts": int(alert.alert_time.timestamp()),
                    }
                ]
            }

            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=10,
            )

            return response.status_code == 200

        except Exception as e:
            print(f"[WARN] Failed to send Slack alert: {e}")
            return False


class AlertManager:
    """
    Manage alert subscriptions and delivery.

    Central coordinator for performance alerts.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.subscriptions: Dict[str, AlertSubscription] = {}
        self.handlers: Dict[AlertChannel, AlertHandler] = {}

        # Initialize default handlers
        self.handlers[AlertChannel.CONSOLE] = ConsoleAlertHandler()
        self.handlers[AlertChannel.FILE] = FileAlertHandler()

        # Alert history
        self.alert_history: List[PerformanceAlert] = []

    def subscribe(
        self,
        strategy_id: str,
        channels: List[AlertChannel],
        severity_filter: Optional[List[str]] = None,
        alert_type_filter: Optional[List[str]] = None,
        min_interval_minutes: int = 15,
    ) -> AlertSubscription:
        """
        Create alert subscription.

        Args:
            strategy_id: Strategy to monitor
            channels: Delivery channels
            severity_filter: Filter by severity
            alert_type_filter: Filter by alert type
            min_interval_minutes: Rate limit interval

        Returns:
            Created subscription
        """
        subscription = AlertSubscription(
            strategy_id=strategy_id,
            channels=channels,
            severity_filter=severity_filter,
            alert_type_filter=alert_type_filter,
            min_interval_minutes=min_interval_minutes,
        )

        self.subscriptions[strategy_id] = subscription

        return subscription

    def unsubscribe(self, strategy_id: str) -> bool:
        """
        Remove subscription.

        Args:
            strategy_id: Strategy ID

        Returns:
            True if removed, False if not found
        """
        if strategy_id in self.subscriptions:
            del self.subscriptions[strategy_id]
            return True

        return False

    def register_handler(self, channel: AlertChannel, handler: AlertHandler) -> None:
        """
        Register custom alert handler.

        Args:
            channel: Channel type
            handler: Handler instance
        """
        self.handlers[channel] = handler

    def send_alert(self, alert: PerformanceAlert) -> int:
        """
        Send alert to subscribed channels.

        Args:
            alert: Performance alert

        Returns:
            Number of channels alert was sent to
        """
        # Find subscription for this strategy
        subscription = self.subscriptions.get(alert.strategy_id)

        if not subscription:
            # No subscription - just log to history
            self.alert_history.append(alert)
            return 0

        # Check if alert should be sent
        if not subscription.should_send(alert):
            return 0

        # Send to all configured channels
        sent_count = 0

        for channel in subscription.channels:
            handler = self.handlers.get(channel)

            if handler:
                try:
                    if handler.send(alert, subscription):
                        sent_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to send alert via {channel.value}: {e}")

        # Add to history
        self.alert_history.append(alert)

        return sent_count

    def send_alerts(self, alerts: List[PerformanceAlert]) -> int:
        """
        Send multiple alerts.

        Args:
            alerts: List of alerts

        Returns:
            Total number of deliveries
        """
        total_sent = 0

        for alert in alerts:
            total_sent += self.send_alert(alert)

        return total_sent

    def get_alert_history(
        self,
        strategy_id: Optional[str] = None,
        hours: Optional[int] = None,
        severity: Optional[str] = None,
    ) -> List[PerformanceAlert]:
        """
        Get alert history with optional filters.

        Args:
            strategy_id: Filter by strategy
            hours: Only alerts from last N hours
            severity: Filter by severity

        Returns:
            Filtered alert history
        """
        filtered = self.alert_history

        if strategy_id:
            filtered = [a for a in filtered if a.strategy_id == strategy_id]

        if hours:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            filtered = [a for a in filtered if a.alert_time >= cutoff]

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        return filtered

    def get_alert_summary(self, hours: int = 24) -> Dict:
        """
        Get summary of recent alerts.

        Args:
            hours: Time window in hours

        Returns:
            Alert summary
        """
        recent = self.get_alert_history(hours=hours)

        summary = {
            "total_alerts": len(recent),
            "critical_alerts": sum(1 for a in recent if a.severity == "critical"),
            "warning_alerts": sum(1 for a in recent if a.severity == "warning"),
            "alerts_by_strategy": {},
            "alerts_by_type": {},
        }

        # Count by strategy
        for alert in recent:
            strategy_id = alert.strategy_id
            summary["alerts_by_strategy"][strategy_id] = summary["alerts_by_strategy"].get(strategy_id, 0) + 1

        # Count by type
        for alert in recent:
            alert_type = alert.alert_type
            summary["alerts_by_type"][alert_type] = summary["alerts_by_type"].get(alert_type, 0) + 1

        return summary

    def clear_history(self, hours: Optional[int] = None) -> int:
        """
        Clear alert history.

        Args:
            hours: Clear alerts older than N hours (None = all)

        Returns:
            Number of alerts cleared
        """
        if hours is None:
            count = len(self.alert_history)
            self.alert_history = []
            return count

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        before_count = len(self.alert_history)
        self.alert_history = [a for a in self.alert_history if a.alert_time >= cutoff]
        after_count = len(self.alert_history)

        return before_count - after_count


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Get global alert manager instance.

    Returns:
        Alert manager singleton
    """
    global _alert_manager

    if _alert_manager is None:
        _alert_manager = AlertManager()

    return _alert_manager


def send_performance_alert(
    strategy_id: str,
    alert_type: str,
    severity: str,
    message: str,
    recommended_action: Optional[str] = None,
) -> int:
    """
    Convenience function to send a performance alert.

    Args:
        strategy_id: Strategy ID
        alert_type: Alert type
        severity: Alert severity ("warning" or "critical")
        message: Alert message
        recommended_action: Recommended action

    Returns:
        Number of channels alert was sent to
    """
    from finantradealgo.research.performance.models import PerformanceMetrics

    alert = PerformanceAlert(
        strategy_id=strategy_id,
        alert_time=datetime.utcnow(),
        alert_type=alert_type,
        severity=severity,
        message=message,
        metrics_snapshot=PerformanceMetrics(),  # Placeholder
        recommended_action=recommended_action,
    )

    manager = get_alert_manager()
    return manager.send_alert(alert)
