import React from 'react';
import styles from '../styles/ExportButton.module.css';

interface ExportButtonProps {
  data: any;
  filename?: string;
  deviceId?: string;
}

export default function ExportButton({ data, filename = 'weather-data', deviceId }: ExportButtonProps) {
  const exportToCSV = () => {
    if (!data) {
      alert('No data available to export');
      return;
    }

    let csvContent = '';
    
    // Handle different data types
    if (data.predictions && data.for_ts) {
      // Forecast data
      csvContent = exportForecastData(data, deviceId);
    } else if (data.device_id && data.ts) {
      // Latest sensor data
      csvContent = exportLatestData(data, deviceId);
    } else if (Array.isArray(data)) {
      // Device list
      csvContent = exportDeviceList(data);
    }

    downloadCSV(csvContent, filename);
  };

  const exportForecastData = (forecast: any, deviceId?: string) => {
    const headers = ['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Wind (m/s)', 'Radiation (W/m²)', 'Precipitation (mm)'];
    const rows = forecast.for_ts.map((timestamp: string, index: number) => [
      new Date(timestamp).toLocaleString(),
      forecast.predictions.temperature?.[index] || '',
      forecast.predictions.humidity?.[index] || '',
      forecast.predictions.wind_speed?.[index] || '',
      forecast.predictions.radiation?.[index] || '',
      forecast.predictions.precipitation?.[index] || ''
    ]);

    return [headers, ...rows].map(row => row.join(',')).join('\n');
  };

  const exportLatestData = (latest: any, deviceId?: string) => {
    const headers = ['Device ID', 'Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Wind (m/s)', 'Radiation (W/m²)', 'Precipitation (mm)'];
    const row = [
      latest.device_id || deviceId || '',
      new Date(latest.ts).toLocaleString(),
      latest.temperature || '',
      latest.humidity || '',
      latest.wind_speed || '',
      latest.radiation || '',
      latest.precipitation || ''
    ];

    return [headers, row].map(row => row.join(',')).join('\n');
  };

  const exportDeviceList = (devices: any[]) => {
    const headers = ['Device ID', 'Last Seen'];
    const rows = devices.map(device => [
      device.device_id,
      device.last_seen ? new Date(device.last_seen).toLocaleString() : 'Never'
    ]);

    return [headers, ...rows].map(row => row.join(',')).join('\n');
  };

  const downloadCSV = (content: string, baseFilename: string) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `${baseFilename}-${new Date().toISOString().split('T')[0]}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <button
      className={styles.exportButton}
      onClick={exportToCSV}
      title="Export data as CSV"
    >
      📊
    </button>
  );
}