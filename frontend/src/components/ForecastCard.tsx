import React, { useMemo } from 'react';
import styles from '../styles/ForecastCard.module.css';
import type { Forecast } from '../api';
import ExportButton from './ExportButton';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ForecastCardProps {
  forecast: Forecast | null;
}

interface ChartDataPoint {
  ts: string;
  temperature?: number;
  humidity?: number;
  wind_speed?: number;
  radiation?: number;
  precipitation?: number;
}

export default function ForecastCard({ forecast }: ForecastCardProps) {
  const chartData = useMemo<ChartDataPoint[]>(() => {
    if (!forecast) return [];
    
    return forecast.for_ts.map((timestamp, index) => ({
      ts: new Date(timestamp).toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
      }),
      temperature: forecast.predictions.temperature?.[index],
      humidity: forecast.predictions.humidity?.[index],
      wind_speed: forecast.predictions.wind_speed?.[index],
      radiation: forecast.predictions.radiation?.[index],
      precipitation: forecast.predictions.precipitation?.[index],
    }));
  }, [forecast]);

  const formatValue = (value: number | undefined, decimals: number, unit: string): string => {
    return value !== undefined ? `${value.toFixed(decimals)}${unit}` : '—';
  };

return (
    <div className={styles.card}>
      <div className={styles.header}>
        <h3>8-hour forecast</h3>
        {forecast && <ExportButton data={forecast} filename={`forecast-${forecast?.device_id}`} />}
      </div>

      <div className={styles.grid}>
        {/* Chart Section */}
        <div className={styles.chartWrap}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ts" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line 
                yAxisId="left" 
                type="monotone" 
                dataKey="temperature" 
                stroke="#ff6b6b"
                dot={false} 
              />
              <Line 
                yAxisId="left" 
                type="monotone" 
                dataKey="humidity" 
                stroke="#4ecdc4"
                dot={false} 
              />
              <Line 
                yAxisId="right" 
                type="monotone" 
                dataKey="wind_speed" 
                stroke="#95e1d3"
                dot={false} 
              />
              <Line 
                yAxisId="right" 
                type="monotone" 
                dataKey="radiation" 
                stroke="#ffd93d"
                dot={false} 
              />
              <Line 
                yAxisId="right" 
                type="monotone" 
                dataKey="precipitation" 
                stroke="#6c5ce7"
                dot={false} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Data Table Section */}
        <div className={styles.vars}>
          {/* Header Row with Symbols */}
          <div className={styles.row} style={{ fontWeight: 600 }}>
            <span className={styles.time}>⏰</span>
            <span>🌡️</span>
            <span>💧</span>
            <span>💨</span>
            <span>☀️</span>
            <span>🌧️</span>
          </div>

          {/* Data Rows */}
          {chartData.map((row, index) => (
            <div key={index} className={styles.row}>
              <span className={styles.time}>{row.ts}</span>
              <span>{formatValue(row.temperature, 1, '°C')}</span>
              <span>{formatValue(row.humidity, 0, '%')}</span>
              <span>{formatValue(row.wind_speed, 1, ' m/s')}</span>
              <span>{formatValue(row.radiation, 0, ' W/m²')}</span>
              <span>{formatValue(row.precipitation, 2, ' mm')}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
