import React, { useEffect, useState } from 'react'
import { getDevices, getLatest, getPredict, Device, Latest, Forecast } from './api'
import styles from './styles/App.module.css'

import DeviceList from './components/DeviceList'
import SensorCard from './components/SensorCard'
import ForecastCard from './components/ForecastCard'
import PredictionText from './components/PredictionText'
import { ThemeProvider, useTheme } from './contexts/ThemeContext'
import LoadingSkeleton from './components/LoadingSkeleton'
import FullscreenButton from './components/FullscreenButton'
import KeyboardShortcuts from './components/KeyboardShortcuts'
import NotificationSettings from './components/NotificationSettings'
import ExportButton from './components/ExportButton'
import CollapsiblePanel from './components/CollapsiblePanel'
import notificationService from './services/notifications'

import { POLL_MS } from './config'

function AppContent() {
  const [devices, setDevices] = useState<Device[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [latest, setLatest] = useState<Latest | null>(null)
  const [forecast, setForecast] = useState<Forecast | null>(null)
  const [loading, setLoading] = useState(true)
  const { isDark, toggleTheme } = useTheme()

  async function refresh(devId: string | null) {
    setLoading(true)
    try {
      const dv = await getDevices()
      setDevices(dv)

      const id = devId ?? (dv[0]?.device_id ?? null)
      if (!id) return

      setSelected(id)

      const lt = await getLatest(id)
      setLatest(lt)

      const fc = await getPredict(id)
      setForecast(fc)
    } finally {
      setLoading(false)
    }
  }

  const handleDeviceChange = (direction: 'next' | 'prev') => {
    if (devices.length === 0) return;
    
    const currentIndex = selected ? devices.findIndex(d => d.device_id === selected) : -1;
    let newIndex;
    
    if (direction === 'next') {
      newIndex = currentIndex + 1 >= devices.length ? 0 : currentIndex + 1;
    } else {
      newIndex = currentIndex - 1 < 0 ? devices.length - 1 : currentIndex - 1;
    }
    
    const newDevice = devices[newIndex];
    if (newDevice) {
      setSelected(newDevice.device_id);
      refresh(newDevice.device_id);
    }
  };

  useEffect(() => {
    // Request notification permission on first load
    notificationService.requestPermission();
    
    refresh(selected)
    const t = setInterval(() => refresh(selected), POLL_MS)
    return () => clearInterval(t)
  }, [selected])

  // Show notification when data updates
  useEffect(() => {
    if (latest && selected && !loading) {
      // Check for threshold alerts
      if (latest.temperature > 35) {
        notificationService.showThresholdAlert('Temperature', latest.temperature, 35);
      }
      if (latest.humidity > 80) {
        notificationService.showThresholdAlert('Humidity', latest.humidity, 80);
      }
      if (latest.pressure < 1000) {
        notificationService.showThresholdAlert('Pressure', latest.pressure, 1000);
      }
      
      // Show data update notification
      notificationService.showDataUpdate(selected);
    }
  }, [latest, selected, loading])

  return (
    <div className={styles.container}>
      <KeyboardShortcuts
        onDeviceChange={handleDeviceChange}
        onThemeToggle={toggleTheme}
        onFullscreen={() => document.documentElement.requestFullscreen()}
        onRefresh={() => refresh(selected)}
      />
      {/* =====================
          NAVBAR
      ===================== */}
      <nav
        style={{
          height: '64px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 28px',
          background: 'rgba(255,255,255,0.7)',
          backdropFilter: 'blur(14px)',
          borderBottom: '1px solid rgba(0,0,0,0.08)',
        }}
      >
        <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>
          🌤 Local Weather Dashboard
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '14px',
            fontSize: '0.9rem',
            color: '#374151',
          }}
        >
          <button
            className={styles.themeToggle}
            onClick={toggleTheme}
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDark ? '☀️' : '🌙'}
          </button>
          <FullscreenButton />
          {selected && <NotificationSettings deviceId={selected} />}
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              background: latest ? '#16a34a' : '#9ca3af',
            }}
          />
          <span>
            {selected ? `Device: ${selected}` : 'No device selected'}
          </span>
        </div>
      </nav>

      {/* =====================
          MAIN LAYOUT
      ===================== */}
      <div className={styles.layout}>
        <aside className={styles.sidebar}>
          {loading ? (
            <LoadingSkeleton type="list" />
          ) : (
            <>
              <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'flex-end' }}>
                <ExportButton data={devices} filename="devices" />
              </div>
              <DeviceList
                devices={devices}
                selected={selected}
                onSelect={setSelected}
              />
            </>
          )}
        </aside>

        <main className={styles.main}>
          {/* =====================
              LIVE SENSOR VALUES
          ===================== */}
          <CollapsiblePanel title="🌡️ Live Sensor Data" defaultOpen={true}>
            <section className={styles.cards}>
              {loading ? (
                <LoadingSkeleton type="card" count={5} />
              ) : (
                <>
                  <div style={{ position: 'relative' }}>
                    <SensorCard title="Temp (°C)" value={latest?.temperature} />
                    {latest && (
                      <div style={{ position: 'absolute', top: '8px', right: '8px' }}>
                        <ExportButton data={latest} filename={`latest-${selected}`} deviceId={selected} />
                      </div>
                    )}
                  </div>
                  <SensorCard title="Humidity (%)" value={latest?.humidity} />
                  <SensorCard title="Wind (m/s)" value={latest?.wind_speed} />
                  <SensorCard title="Radiation (W/m²)" value={latest?.radiation} />
                  <SensorCard title="Precip (mm)" value={latest?.precipitation} />
                </>
              )}
            </section>
          </CollapsiblePanel>

          {/* =====================
              NUMERIC FORECAST
          ===================== */}
          <CollapsiblePanel title="📈 8-Hour Forecast" defaultOpen={true}>
            {loading ? <LoadingSkeleton type="chart" /> : <ForecastCard forecast={forecast} />}
          </CollapsiblePanel>

          {/* =====================
              TEXT PREDICTIONS
          ===================== */}
          <CollapsiblePanel title="🔮 Weather Insights" defaultOpen={false}>
            {loading ? (
              <LoadingSkeleton type="text" />
            ) : (
              selected && <PredictionText deviceId={selected} />
            )}
          </CollapsiblePanel>
        </main>
      </div>

      {/* =====================
          FOOTER
      ===================== */}
      <footer className={styles.footer}>
        <small>
          Polling every {(POLL_MS / 1000 / 60).toFixed(0)} min • Model:{' '}
          {forecast?.model_version ?? '—'}
        </small>
      </footer>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  )
}
