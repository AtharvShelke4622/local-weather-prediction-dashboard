import React from 'react';
import styles from '../styles/LoadingSkeleton.module.css';

interface LoadingSkeletonProps {
  type?: 'card' | 'text' | 'chart' | 'list';
  count?: number;
}

export default function LoadingSkeleton({ type = 'card', count = 1 }: LoadingSkeletonProps) {
  const renderSkeleton = () => {
    switch (type) {
      case 'card':
        return (
          <div className={styles.skeletonCard}>
            <div className={styles.skeletonHeader}></div>
            <div className={styles.skeletonValue}></div>
          </div>
        );
      case 'text':
        return (
          <div className={styles.skeletonText}>
            <div className={styles.skeletonLine}></div>
            <div className={`${styles.skeletonLine} ${styles.short}`}></div>
          </div>
        );
      case 'chart':
        return (
          <div className={styles.skeletonChart}>
            <div className={styles.skeletonChartArea}></div>
            <div className={styles.skeletonChartLegend}></div>
          </div>
        );
      case 'list':
        return (
          <div className={styles.skeletonList}>
            {[...Array(3)].map((_, i) => (
              <div key={i} className={styles.skeletonListItem}></div>
            ))}
          </div>
        );
      default:
        return <div className={styles.skeletonDefault}></div>;
    }
  };

  return (
    <>
      {[...Array(count)].map((_, index) => (
        <div key={index}>{renderSkeleton()}</div>
      ))}
    </>
  );
}