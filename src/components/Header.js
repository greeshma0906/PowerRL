import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <h1 className="header-title"> PowerRL</h1>
        <p className="header-subtitle">Interactive Energy Tracking for your RL Pipelines</p>
      </div>
    </header>
  );
}

export default Header;
