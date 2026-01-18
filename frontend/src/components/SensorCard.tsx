import React from 'react';
import styles from '../styles/SensorCard.module.css';

interface SensorCardProps {
  title: string;
  value?: number;
  unit?: string;
  decimals?: number;
}

export default function SensorCard({ 
  title, 
  value, 
  unit = '', 
  decimals = 1 
}: SensorCardProps) {
  const formatValue = (): string => {
    if (value === undefined || value === null) return '—';
    return `${value.toFixed(decimals)}${unit}`;
  };

  const getWeatherIcon = (title: string): string => {
    const icons: Record<string, string> = {
      'Temp (°C)': '🌡️',
      'Humidity (%)': '💧',
      'Wind (m/s)': '💨',
      'Radiation (W/m²)': '☀️',
      'Precip (mm)': '🌧️',
      'Temperature': '🌡️',
      'Humidity': '💧',
      'Wind Speed': '💨',
      'Solar Radiation': '☀️',
      'Precipitation': '🌧️'
    };
    return icons[title] || '📊';
  };

  return (
    <div className={styles.card}>
      <div className={styles.title}>
        <span className={styles.icon}>{getWeatherIcon(title)}</span>
        {title}
      </div>
      <div className={styles.value}>{formatValue()}</div>
    </div>
  );
}