import axios from 'axios'
import { API_BASE } from './config'

export const api = axios.create({
  baseURL: API_BASE
})

export type Device = { device_id: string; last_seen: string | null }
export type Latest = {
    device_id: string; ts: string;
    temperature?: number;
    humidity?: number;
    wind_speed?: number;
    radiation?: number;
    precipitation?: number;
}
export type Forecast = {
    device_id: string;
    pred_ts: string[];
    predictions: Record<string, number[]>;
    model_version: string;
    prediction_text?: Record<string, string[]>;
}
export type Forecast = {
  device_id: string; pred_ts: string; for_ts: string[];
  predictions: Record<string, number[]>; model_version: string;
  prediction_text?: Record<string, string[]>; // Optional prediction text descriptions
}

export async function getDevices(): Promise<Device[]> {
  const { data } = await api.get('/api/v1/devices')
  return data
}
export async function getLatest(device_id: string): Promise<Latest | null> {
  const { data } = await api.get('/api/v1/latest', { params: { device_id } })
  return data
}
export async function getPredict(device_id: string): Promise<Forecast | null> {
  const { data } = await api.get('/api/v1/predict', { params: { device_id } })
  return data
}
