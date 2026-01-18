import React, { useState } from 'react';
import styles from '../styles/CollapsiblePanel.module.css';

interface CollapsiblePanelProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  className?: string;
}

export default function CollapsiblePanel({ 
  title, 
  children, 
  defaultOpen = true, 
  className = '' 
}: CollapsiblePanelProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const togglePanel = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={`${styles.panel} ${className} ${isOpen ? styles.open : styles.closed}`}>
      <div className={styles.header} onClick={togglePanel}>
        <h3 className={styles.title}>{title}</h3>
        <button className={styles.toggleButton} aria-expanded={isOpen}>
          <span className={`${styles.arrow} ${isOpen ? styles.arrowUp : styles.arrowDown}`}>
            ▼
          </span>
        </button>
      </div>
      
      <div className={`${styles.content} ${isOpen ? styles.contentOpen : styles.contentClosed}`}>
        {children}
      </div>
    </div>
  );
}