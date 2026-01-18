import notificationIcons from '../assets/notification-icons';

interface NotificationOptions {
  title: string;
  body: string;
  icon?: string;
  tag?: string;
  requireInteraction?: boolean;
}

class NotificationService {
  private permission: NotificationPermission = 'default';
  private isEnabled: boolean = false;

  constructor() {
    this.checkPermission();
  }

  private checkPermission() {
    if ('Notification' in window) {
      this.permission = Notification.permission;
      this.isEnabled = this.permission === 'granted';
    }
  }

  async requestPermission(): Promise<boolean> {
    if ('Notification' in window) {
      if (this.permission === 'default') {
        const permission = await Notification.requestPermission();
        this.permission = permission;
        this.isEnabled = permission === 'granted';
      }
      return this.isEnabled;
    }
    return false;
  }

  show(options: NotificationOptions) {
    if (!this.isEnabled) {
      console.warn('Notifications not enabled. Call requestPermission() first.');
      return;
    }

    if ('Notification' in window) {
      const notification = new Notification(options.title, {
        body: options.body,
        icon: options.icon || notificationIcons.weather,
        tag: options.tag,
        requireInteraction: options.requireInteraction || false,
        silent: false,
        vibrate: [200, 100, 200],
      });

      // Auto-close after 5 seconds unless requireInteraction is true
      if (!options.requireInteraction) {
        setTimeout(() => {
          notification.close();
        }, 5000);
      }

      // Handle click events
      notification.onclick = () => {
        window.focus();
        notification.close();
      };

      // Handle error events
      notification.onerror = (error) => {
        console.error('Notification error:', error);
      };

      // Handle close events
      notification.onclose = () => {
        console.log('Notification closed');
      };

      return notification;
    }
  }

  // Weather-specific notification methods
  showWeatherAlert(deviceId: string, message: string) {
    this.show({
      title: `🌤️ Weather Alert - ${deviceId}`,
      body: message,
      icon: notificationIcons.alert,
      tag: `weather-alert-${deviceId}`,
      requireInteraction: true,
    });
  }

  showDeviceOffline(deviceId: string) {
    this.show({
      title: '🔴 Device Offline',
      body: `Device ${deviceId} is no longer reporting data`,
      icon: notificationIcons.offline,
      tag: `device-offline-${deviceId}`,
      requireInteraction: true,
    });
  }

  showDataUpdate(deviceId: string) {
    this.show({
      title: '🔄 Data Updated',
      body: `New weather data available for ${deviceId}`,
      icon: notificationIcons.update,
      tag: `data-update-${deviceId}`,
    });
  }

  showThresholdAlert(metric: string, value: number, threshold: number) {
    this.show({
      title: '⚠️ Threshold Alert',
      body: `${metric} (${value}) exceeds threshold (${threshold})`,
      icon: notificationIcons.threshold,
      tag: `threshold-${metric}`,
      requireInteraction: true,
    });
  }

  showDeviceOnline(deviceId: string) {
    this.show({
      title: '🟢 Device Online',
      body: `Device ${deviceId} is now reporting data`,
      icon: notificationIcons.online,
      tag: `device-online-${deviceId}`,
    });
  }

  showWeatherWarning(deviceId: string, condition: string, severity: 'low' | 'medium' | 'high') {
    const icons = {
      low: '🟡',
      medium: '🟠', 
      high: '🔴'
    };
    
    this.show({
      title: `${icons[severity]} Weather Warning - ${deviceId}`,
      body: `${condition} detected. Severity: ${severity}`,
      icon: notificationIcons.warning,
      tag: `weather-warning-${deviceId}`,
      requireInteraction: severity === 'high',
    });
  }

  showSystemMessage(message: string, type: 'info' | 'warning' | 'error' = 'info') {
    const icons = {
      info: 'ℹ️',
      warning: '⚠️',
      error: '❌'
    };
    
    this.show({
      title: `${icons[type]} System Message`,
      body: message,
      icon: notificationIcons.system,
      tag: 'system-message',
      requireInteraction: type === 'error',
    });
  }

  isSupported(): boolean {
    return 'Notification' in window;
  }

  getPermissionStatus(): NotificationPermission {
    return this.permission;
  }
}

export const notificationService = new NotificationService();
export default notificationService;