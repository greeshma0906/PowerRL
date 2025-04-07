import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function ModelEmissions() {
  // Data for the emissions chart
  const chartData = {
    labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'],
    datasets: [
      {
        label: 'Carbon Emissions (CO2 lbs)',
        data: [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.7, 2.9],
        borderColor: 'rgba(75, 192, 75, 1)',
        backgroundColor: 'rgba(75, 192, 75, 0.2)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
      },
      {
        label: 'Alternative Consumption',
        data: [0, 0.8, 1.5, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.3, 6.7, 7.0, 7.2],
        borderColor: 'rgba(255, 206, 86, 1)',
        backgroundColor: 'rgba(255, 206, 86, 0.2)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
      },
    ],
  };

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
          text: 'Epochs (10 = extrapolated)',
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
        <h3>Your Model's CO<sub>2</sub> Emissions (CO<sub>2</sub> lbs)</h3>
        <button className="settings-btn">
          <i className="fa fa-cog"></i>
        </button>
      </div>
      
      <div className="chart-wrapper">
        <Line data={chartData} options={chartOptions} height={300} />
      </div>
      
      <div className="chart-legend">
        <div className="legend-item">
          <div className="color-box green"></div>
          <span>Carbon Emissions (CO2 lbs)</span>
        </div>
        <div className="legend-item">
          <div className="color-box yellow"></div>
          <span>Alternative Consumption</span>
        </div>
      </div>
      
      <div className="emissions-explanation">
        <h4>How Your CO<sub>2</sub> Emissions are Calculated</h4>
      </div>
    </div>
  );
}

export default ModelEmissions;
