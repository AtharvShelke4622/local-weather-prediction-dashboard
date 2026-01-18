import { useEffect, useCallback } from 'react';
import { useAuthStore } from '../stores/authStore';

// Custom hook for API calls with authentication
export const useAuthenticatedApi = () => {
  const { token, isAuthenticated } = useAuthStore();

  const apiCall = useCallback(async (endpoint: string, options: RequestInit = {}) => {
    if (!isAuthenticated || !token) {
      throw new Error('Not authenticated');
    }

    const url = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    };

    const mergedOptions = {
      ...defaultOptions,
      ...options,
      headers: {
        ...defaultOptions.headers,
        ...options.headers,
      },
    };

    const response = await fetch(url, mergedOptions);

    if (response.status === 401) {
      // Token expired or invalid, logout user
      useAuthStore.getState().logout();
      throw new Error('Session expired');
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || response.statusText);
    }

    return response.json();
  }, [token, isAuthenticated]);

  return { apiCall, isAuthenticated };
};

// Custom hook for automatic data refresh
export const useAutoRefresh = (
  refreshFunction: () => Promise<void>,
  intervalMs: number = 300000, // 5 minutes default
  dependencies: any[] = []
) => {
  useEffect(() => {
    if (refreshFunction) {
      // Initial refresh
      refreshFunction();

      // Set up interval for auto-refresh
      const interval = setInterval(refreshFunction, intervalMs);

      return () => clearInterval(interval);
    }
  }, [refreshFunction, intervalMs, ...dependencies]);
};

// Custom hook for error handling
export const useErrorHandler = () => {
  const { clearError } = useAuthStore();

  const handleError = useCallback((error: Error, context?: string) => {
    console.error(`Error${context ? ` in ${context}` : ''}:`, error);
    
    // You could integrate with a toast system here
    // toast.error(error.message);
  }, []);

  const clearErrors = useCallback(() => {
    clearError();
  }, [clearError]);

  return { handleError, clearErrors };
};