import React from 'react';
import styles from '../styles/Navbar.module.css';

type PageType = 'dashboard' | 'forecast' | 'insights';

interface NavbarProps {
  active: PageType;
  onNavigate: (page: PageType) => void;
}

interface NavLink {
  id: PageType;
  label: string;
}

const NAV_LINKS: NavLink[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'forecast', label: 'Forecast' },
  { id: 'insights', label: 'Insights' },
];

export default function Navbar({ active, onNavigate }: NavbarProps) {
  return (
    <nav className={styles.navbar}>
      <div className={styles.brand}>Local Weather Dashboard</div>

      <div className={styles.links}>
        {NAV_LINKS.map((link) => (
          <button
            key={link.id}
            className={active === link.id ? styles.active : ''}
            onClick={() => onNavigate(link.id)}
            aria-current={active === link.id ? 'page' : undefined}
          >
            {link.label}
          </button>
        ))}
      </div>
    </nav>
  );
}