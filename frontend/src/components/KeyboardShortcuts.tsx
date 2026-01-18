import React, { useEffect } from 'react';

interface KeyboardShortcutsProps {
  onDeviceChange: (direction: 'next' | 'prev') => void;
  onThemeToggle: () => void;
  onFullscreen: () => void;
  onRefresh: () => void;
}

export default function KeyboardShortcuts({
  onDeviceChange,
  onThemeToggle,
  onFullscreen,
  onRefresh,
}: KeyboardShortcutsProps) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Ignore when user is typing in input fields
      if ((event.target as HTMLElement).tagName === 'INPUT' || 
          (event.target as HTMLElement).tagName === 'TEXTAREA') {
        return;
      }

      // Prevent default for our shortcuts
      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case 'r':
            event.preventDefault();
            onRefresh();
            break;
          case 'd':
            event.preventDefault();
            onThemeToggle();
            break;
          case 'f':
            event.preventDefault();
            onFullscreen();
            break;
        }
      } else {
        switch (event.key) {
          case 'ArrowRight':
          case 'n':
            event.preventDefault();
            onDeviceChange('next');
            break;
          case 'ArrowLeft':
          case 'p':
            event.preventDefault();
            onDeviceChange('prev');
            break;
          case 'h':
            event.preventDefault();
            // Show help
            alert(`Keyboard Shortcuts:
            
Navigation:
  → or n - Next device
  ← or p - Previous device
  
Controls:
  Ctrl/Cmd + R - Refresh data
  Ctrl/Cmd + D - Toggle dark mode
  Ctrl/Cmd + F - Toggle fullscreen
  H - Show this help`);
            break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onDeviceChange, onThemeToggle, onFullscreen, onRefresh]);

  return null; // This component doesn't render anything
}