import React, { useState, useEffect } from 'react';
import notificationService from '../services/notifications';
import styles from '../styles/NotificationSettings.module.css';

interface NotificationSettingsProps {
  deviceId: string;
}

export default function NotificationSettings({ deviceId }: NotificationSettingsProps) {
  const [isEnabled, setIsEnabled] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    setIsEnabled(notificationService.getPermissionStatus() === 'granted');
  }, []);

  const handleEnableNotifications = async () => {
    const granted = await notificationService.requestPermission();
    setIsEnabled(granted);
    if (granted) {
      notificationService.showWeatherAlert(deviceId, 'Notifications enabled successfully!');
    }
  };

  const testNotification = () => {
    notificationService.showWeatherAlert(deviceId, 'This is a test weather alert notification');
  };

  const handleThresholdAlert = () => {
    notificationService.showThresholdAlert('Temperature', 38, 35);
  };

  const testSystemMessage = () => {
    notificationService.showSystemMessage('System is running normally', 'info');
  };

  if (!showSettings) {
    return (
      <button
        className={styles.settingsButton}
        onClick={() => setShowSettings(true)}
        title="Notification settings"
      >
        🔔
      </button>
    );
  }

  return (
    <div className={styles.settingsPanel}>
      <div className={styles.settingsHeader}>
        <h3>Notification Settings</h3>
        <button
          className={styles.closeButton}
          onClick={() => setShowSettings(false)}
        >
          ✕
        </button>
      </div>
      
      <div className={styles.settingsContent}>
        <div className={styles.settingItem}>
          <label className={styles.settingLabel}>
            <input
              type="checkbox"
              checked={isEnabled}
              onChange={handleEnableNotifications}
            />
            Enable browser notifications
          </label>
          {!isEnabled && notificationService.isSupported() && (
            <p className={styles.helpText}>
              Click to enable notifications for weather alerts and data updates
            </p>
          )}
          {!notificationService.isSupported() && (
            <p className={styles.errorText}>
              Your browser doesn't support notifications
            </p>
          )}
        </div>

        {isEnabled && (
          <>
            <div className={styles.settingItem}>
              <h4>Test Notifications:</h4>
              <div className={styles.buttonGroup}>
                <button
                  className={styles.testButton}
                  onClick={testNotification}
                >
                  Test Weather Alert
                </button>
                <button
                  className={styles.testButton}
                  onClick={handleThresholdAlert}
                >
                  Test Threshold Alert
                </button>
                <button
                  className={styles.testButton}
                  onClick={testSystemMessage}
                >
                  Test System Message
                </button>
              </div>
            </div>

            <div className={styles.settingItem}>
              <h4>Notification Types:</h4>
              <ul className={styles.featureList}>
                <li>🔄 Data updates when new readings arrive</li>
                <li>⚠️ Threshold alerts for extreme weather (temp &gt; 35°C, humidity &gt; 80%)</li>
                <li>📱 Device offline/online status changes</li>
                <li>🌡️ Weather condition changes and warnings</li>
                <li>ℹ️ System messages and status updates</li>
                <li>🔔 Rich notifications with icons and vibration</li>
              </ul>
            </div>
          </>
        )}
      </div>
    </div>
  );
}