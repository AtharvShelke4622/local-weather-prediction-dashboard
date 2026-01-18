/**
 * Configuration file for frontend
 */
import axios from 'axios';

export const POLL_MS = Number(import.meta.env.VITE_POLL_MS) || 300000;
export const API_BASE = 'http://localhost:8000';

// API Base URL configuration
export const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
});

// Device types matching backend schemas
export type Device = {
  device_id: string;
  last_seen: string | null;
};

export type Latest = {
  device_id: string;
  ts: string;
  temperature?: number;
  humidity?: number;
  wind_speed?: number;
  radiation?: number;
  precipitation?: number;
};

export type Forecast = {
  device_id: string;
  pred_ts: string[];
  predictions: Record<string, number[]>;
  model_version: string;
  prediction_text?: Record<string, string[]>;
};

// API functions
export const getDevices = async (): Promise<Device[]> => {
  try {
    const response = await api.get('/api/v1/devices');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch devices:', error);
    return [];
  }
};

export const getLatest = async (device_id: string): Promise<Latest | null> => {
    try {
    const response = await api.get('/api/v1/latest', { params: { device_id } });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch latest:', error);
    return null;
  }
};

export const getPredict = async (device_id: string): Promise<Forecast | null> => {
  try {
    const response = await api.get('/api/v1/predict', { params: { device_id } });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch predictions:', error);
    return null;
  }
};

export default { api };