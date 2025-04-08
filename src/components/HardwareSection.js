import React, { useState, useEffect } from 'react';
import './HardwareSection.css';

function HardwareSection({ initialHardware = [] }) {
  // Use safe fallback
  const hardwareList = Array.isArray(initialHardware) ? initialHardware : [];
  const [hardware, setHardware] = useState(
    hardwareList.length > 0
      ? hardwareList
      : [
          { id: 1, type: 'CPU', model: 'Intel i7', quantity: 1 },
          { id: 2, type: 'GPU', model: 'GTX-1080-Ti', quantity: 1 }
        ]
  );

  const [selectedHardware, setSelectedHardware] = useState('');

  const hardwareOptions = [
    { value: 'cpu-i9-12900k', label: 'CPU - Intel i9 12900K' },
    { value: 'cpu-ryzen-9-5950x', label: 'CPU - AMD Ryzen 9 5950X' },
    { value: 'gpu-rtx-3090', label: 'GPU - NVIDIA RTX 3090' },
    { value: 'gpu-rtx-3080', label: 'GPU - NVIDIA RTX 3080' },
    { value: 'gpu-radeon-rx-6900xt', label: 'GPU - AMD Radeon RX 6900XT' }
  ];

  const addHardware = () => {
    if (!selectedHardware) return;

    const selected = hardwareOptions.find(option => option.value === selectedHardware);

    if (selected) {
      const [type, model] = selected.label.split(' - ');
      const newHardware = {
        id: hardware.length + 1,
        type,
        model,
        quantity: 1,
      };

      setHardware(prev => [...prev, newHardware]);
      setSelectedHardware('');
    }
  };

  const updateQuantity = (id, increase) => {
    setHardware(prev =>
      prev.map(item =>
        item.id === id
          ? { ...item, quantity: increase ? item.quantity + 1 : Math.max(0, item.quantity - 1) }
          : item
      )
    );
  };

  return (
    <div className="hardware-section">
      <h3>Your Hardware</h3>

      <div className="hardware-list">
        {hardware.length > 0 ? (
          hardware.map(item => (
            <div key={item.id} className="hardware-item">
              <div className="hardware-icon">
                {item.type === 'CPU' ? '🔲' : '📊'}
              </div>
              <div className="hardware-details">
                <div className="hardware-name">
                  {item.type} - {item.model}
                </div>
                <div className="hardware-quantity">
                  Quantity: {item.quantity}
                  {item.alternative && (
                    <span className="alternative-badge">Alternative: {item.alternative}</span>
                  )}
                </div>
              </div>
              <div className="quantity-controls">
                <button className="quantity-btn" onClick={() => updateQuantity(item.id, true)}>+</button>
                <button className="quantity-btn" onClick={() => updateQuantity(item.id, false)}>-</button>
              </div>
            </div>
          ))
        ) : (
          <p>Loading hardware data...</p>
        )}
      </div>

      <div className="add-hardware">
        <h4>Add Alternative Hardware</h4>
        <div className="add-hardware-controls">
          <select 
            value={selectedHardware}
            onChange={(e) => setSelectedHardware(e.target.value)}
            className="hardware-select"
          >
            <option value="">Select Hardware</option>
            {hardwareOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <button 
            className="add-btn"
            onClick={addHardware}
            disabled={!selectedHardware}
          >
            ADD
          </button>
        </div>
      </div>
    </div>
  );
}

export default HardwareSection;
	
