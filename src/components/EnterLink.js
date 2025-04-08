
import React, { useState } from 'react';
import './EnterLink.css';
import ModelEmissions from './ModelEmission';

function EnterLink() {
  const [url, setUrl] = useState('');
  const [chartData, setChartData] = useState(null);

  const handleFetchData = async () => {
    try {
      const response = await fetch(`${url}/energy-stats`);
      const json = await response.json();
      console.log(json)
      const epochs = json.map((entry, index) => index.toString());
      const emissions = json.map(entry => entry.energy);

      const data = {
        labels: epochs,
        datasets: [
          {
            label: 'Carbon Emissions (CO2 lbs)',
            data: emissions,
            borderColor: 'rgba(75, 192, 75, 1)',
            backgroundColor: 'rgba(75, 192, 75, 0.2)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
          }
        ],
      };

      setChartData(data);
    } catch (err) {
      console.error('Error fetching data:', err);
    }
  };

  return (
    <div className="enter-link-container">
      <h3 className="enter-link-heading">Enter the link to your RL pipeline:</h3>
      <div className="enter-link-bar">
        <input
          type="text"
          placeholder="Paste your link here..."
          className="link-input"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button className="measure-button" onClick={handleFetchData}>
          Measure
        </button>
      </div>

      {chartData && <ModelEmissions chartData={chartData} />}
    </div>
  );
}

export default EnterLink;

   
