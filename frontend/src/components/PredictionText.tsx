import React, { useEffect, useState } from 'react';
import { fetchPredictionText } from '../predictions';
import styles from '../styles/PredictionCard.module.css';

interface PredictionTextProps {
  deviceId: string;
}

interface PredictionTextResponse {
  prediction_text: Record<string, string[]>;
}

export default function PredictionText({ deviceId }: PredictionTextProps) {
  const [data, setData] = useState<PredictionTextResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!deviceId) {
      setLoading(false);
      return;
    }

    const loadPredictions = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetchPredictionText(deviceId);
        setData(response);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load predictions';
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    loadPredictions();
  }, [deviceId]);

  const formatMetricTitle = (metric: string): string => {
    return metric.replace(/_/g, ' ');
  };

  if (loading) {
    return <div className={styles.loading}>Loading prediction insights…</div>;
  }

  if (error) {
    return <div className={styles.error}>Error: {error}</div>;
  }

  if (!data || Object.keys(data.prediction_text).length === 0) {
    return <div className={styles.empty}>No predictions available.</div>;
  }

  return (
    <div className={styles.predictionText}>
      <h2 className={styles.title}>Weather Insights</h2>

      {Object.entries(data.prediction_text).map(([metric, messages]) => (
        <div
          key={metric}
          className={`${styles.block} ${styles[`metric_${metric}`] ?? ''}`}
        >
          <h3 className={styles.metricTitle}>
            {formatMetricTitle(metric)}
          </h3>

          <ul className={styles.list}>
            {messages.map((message, index) => (
              <li key={index} className={styles.item}>
                {message}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}