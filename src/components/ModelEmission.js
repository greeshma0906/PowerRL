

import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend);

function ModelEmissions({ chartData, title, isEpochData }) {
  // Set up different scale configurations for interval vs epoch data
  const options = {
    responsive: true,
    plugins: {
      legend: { display: true },
      title: { display: true, text: title || 'Energy(kWH)' },
    },
    scales: {
      x: {
        type: 'category',
        labels: isEpochData
          ? ['1', '2','3', '4', '5', '6', '7', '9', '10'] // for epoch data
          : undefined, // interval data uses default
        title: {
          display: true,
          text: isEpochData ? 'Training Epoch' : 'Interval',
        },
        ticks: {
          autoSkip: true,
          maxRotation: 0,
          minRotation: 0,
        },
      },
      y: {
        type: 'linear',
        title: { display: true, text: 'Energy(kWH)' },
      },
    },
  };

  return (
    <div style={{ maxWidth: '700px', margin: '20px auto' }}>
      <Line data={chartData} options={options} />
    </div>
  );
}

export default ModelEmissions;


