import React from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
import { Heatmap } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, Tooltip, Legend);

function ResourceHeatmap() {
  const data = {
    labels: ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'],
    datasets: [
      {
        label: 'CPU Usage (%)',
        data: [40, 55, 70, 85, 60],
        backgroundColor: '#FF9800',
      },
      {
        label: 'GPU Usage (%)',
        data: [30, 50, 80, 90, 75],
        backgroundColor: '#FFCC80',
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
    },
    scales: {
      x: { title: { display: true, text: 'Pipeline Stages' } },
      y: { title: { display: true, text: 'Usage (%)' }, min: 0, max: 100 },
    },
  };

  return (
    <div>
      <h3>Resource Usage Heatmap</h3>
      <Heatmap data={data} options={options} />
    </div>
  );
}

export default ResourceHeatmap;
