import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

export interface Device {
  device_id: string;
  last_seen?: string;
}

export interface SensorReading {
  device_id: string;
  ts: string;
  temperature?: number;
  humidity?: number;
  wind_speed?: number;
  radiation?: number;
  precipitation?: number;
}

export interface ForecastData {
  device_id: string;
  pred_ts: string;
  for_ts: string[];
  predictions: {
    temperature: number[];
    humidity: number[];
    wind_speed: number[];
    radiation: number[];
    precipitation: number[];
  };
  model_version: string;
}

export interface WeatherState {
  // Device management
  devices: Device[];
  selectedDevice: string | null;
  
  // Sensor readings
  latestReading: SensorReading | null;
  historicalData: SensorReading[];
  
  // Forecasts
  forecast: ForecastData | null;
  isGeneratingForecast: boolean;
  
  // UI state
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
  
  // Actions
  fetchDevices: () => Promise<void>;
  selectDevice: (deviceId: string) => void;
  fetchLatestReading: (deviceId: string) => Promise<void>;
  fetchHistoricalData: (deviceId: string, hours?: number) => Promise<void>;
  generateForecast: (deviceId: string) => Promise<void>;
  clearError: () => void;
  refreshData: () => Promise<void>;
}

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Helper function for authenticated API calls
const apiCall = async (endpoint: string, token: string | null) => {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'GET',
    headers,
  });

  if (!response.ok) {
    throw new Error(`API call failed: ${response.statusText}`);
  }

  return response.json();
};

export const useWeatherStore = create<WeatherState>()(
  subscribeWithSelector((set, get) => ({
    devices: [],
    selectedDevice: null,
    latestReading: null,
    historicalData: [],
    forecast: null,
    isGeneratingForecast: false,
    isLoading: false,
    error: null,
    lastUpdated: null,

    fetchDevices: async () => {
      set({ isLoading: true, error: null });
      
      try {
        // Get token from auth store
        const authStore = JSON.parse(localStorage.getItem('auth-storage') || '{}');
        const token = authStore.state?.token;
        
        if (!token) {
          throw new Error('Not authenticated');
        }

        const devices = await apiCall('/api/v1/devices', token);
        
        set({
          devices,
          isLoading: false,
          lastUpdated: new Date().toISOString(),
        });
      } catch (error) {
        set({
          error: error instanceof Error ? error.message : 'Failed to fetch devices',
          isLoading: false,
        });
      }
    },

    selectDevice: (deviceId: string) => {
      set({ selectedDevice: deviceId });
      // Auto-fetch data for selected device
      get().fetchLatestReading(deviceId);
      get().generateForecast(deviceId);
    },

    fetchLatestReading: async (deviceId: string) => {
      set({ isLoading: true, error: null });
      
      try {
        const authStore = JSON.parse(localStorage.getItem('auth-storage') || '{}');
        const token = authStore.state?.token;
        
        if (!token) {
          throw new Error('Not authenticated');
        }

        const reading = await apiCall(`/api/v1/latest?device_id=${encodeURIComponent(deviceId)}`, token);
        
        set({
          latestReading: reading,
          isLoading: false,
        });
      } catch (error) {
        set({
          error: error instanceof Error ? error.message : 'Failed to fetch latest reading',
          isLoading: false,
        });
      }
    },

    fetchHistoricalData: async (deviceId: string, hours: number = 24) => {
      set({ isLoading: true, error: null });
      
      try {
        const authStore = JSON.parse(localStorage.getItem('auth-storage') || '{}');
        const token = authStore.state?.token;
        
        if (!token) {
          throw new Error('Not authenticated');
        }

        // This would need a new endpoint for historical data
        // For now, we'll use a placeholder
        const historicalData = await apiCall(`/api/v1/historical?device_id=${encodeURIComponent(deviceId)}&hours=${hours}`, token);
        
        set({
          historicalData,
          isLoading: false,
        });
      } catch (error) {
        set({
          error: error instanceof Error ? error.message : 'Failed to fetch historical data',
          isLoading: false,
        });
      }
    },

    generateForecast: async (deviceId: string) => {
      set({ isGeneratingForecast: true, error: null });
      
      try {
        const authStore = JSON.parse(localStorage.getItem('auth-storage') || '{}');
        const token = authStore.state?.token;
        
        if (!token) {
          throw new Error('Not authenticated');
        }

        const forecast = await apiCall(`/api/v1/predict?device_id=${encodeURIComponent(deviceId)}`, token);
        
        set({
          forecast,
          isGeneratingForecast: false,
        });
      } catch (error) {
        set({
          error: error instanceof Error ? error.message : 'Failed to generate forecast',
          isGeneratingForecast: false,
        });
      }
    },

    clearError: () => {
      set({ error: null });
    },

    refreshData: async () => {
      const { selectedDevice } = get();
      if (selectedDevice) {
        await Promise.all([
          get().fetchLatestReading(selectedDevice),
          get().generateForecast(selectedDevice),
        ]);
      }
    },
  }))
);