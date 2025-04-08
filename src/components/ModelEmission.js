
import React from 'react';
import { Line } from 'react-chartjs-2';

function ModelEmissions({ chartData }) {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Carbon Emissions (CO2 lbs)',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Epochs',
        },
      },
    },
    plugins: {
      legend: {
        position: 'top',
        align: 'start',
        labels: {
          boxWidth: 15,
          usePointStyle: false,
        },
      },
    },
  };

  return (
    <div className="emissions-chart-container">
      <div className="section-header">
        <h3>Your Model's CO<sub>2</sub> Emissions</h3>
      </div>

      <div className="chart-wrapper">
        {chartData ? (
          <Line data={chartData} options={chartOptions} height={300} />
        ) : (
          <p>Loading chart...</p>
        )}
      </div>
    </div>
  );
}

export default ModelEmissions;


