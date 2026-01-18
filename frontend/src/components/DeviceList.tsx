import React from 'react';
import styles from '../styles/DeviceList.module.css';
import type { Device } from '../api';

interface DeviceListProps {
  devices: Device[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function DeviceList({ devices, selected, onSelect }: DeviceListProps) {
  const formatLastSeen = (lastSeen: string | null | undefined): string => {
    if (!lastSeen) return 'never';
    return new Date(lastSeen).toLocaleString();
  };

  return (
    <div className={styles.wrapper}>
      <h3 className={styles.title}>Devices</h3>
      <ul className={styles.list}>
        {devices.map((device) => {
          const isSelected = selected === device.device_id;
          
          return (
            <li key={device.device_id}>
              <button
                className={isSelected ? styles.active : styles.item}
                onClick={() => onSelect(device.device_id)}
                aria-pressed={isSelected}
              >
                <span className={styles.deviceId}>{device.device_id}</span>
                <span className={styles.lastSeen}>
                  {formatLastSeen(device.last_seen)}
                </span>
              </button>
            </li>
          );
        })}
        {devices.length === 0 && (
          <li className={styles.empty}>No devices yet</li>
        )}
      </ul>
    </div>
  );
}