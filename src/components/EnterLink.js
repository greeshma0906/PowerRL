import React, { useState } from "react";
import "./EnterLink.css";
import ModelEmissions from "./ModelEmission";
import HardwareSection from "./HardwareSection";

function EnterLink() {
  const [url, setUrl] = useState("");
  const [chartData, setChartData] = useState(null);
  const [hardwareData, setHardwareData] = useState([]);

  const handleFetchData = async () => {
    try {
      // Fetch emissions data
      const energyRes = await fetch(`${url}/energy-stats`);
      const energyJson = await energyRes.json();

      const epochs = energyJson.map((entry, index) => index.toString());
      const emissions = energyJson.map((entry) => entry.energy);

      const data = {
        labels: epochs,
        datasets: [
          {
            label: "Carbon Emissions (CO2 lbs)",
            data: emissions,
            borderColor: "rgba(75, 192, 75, 1)",
            backgroundColor: "rgba(75, 192, 75, 0.2)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
          },
        ],
      };
      setChartData(data);

      // Fetch hardware setup data
      const setupRes = await fetch(`${url}/initial-setup`);
      const setupJson = await setupRes.json();

      const hwList = [];

      const cpu = setupJson.component_names?.cpu;
      if (cpu) {
        for (const model in cpu) {
          hwList.push({
            id: hwList.length + 1,
            type: "CPU",
            model: model,
            quantity: cpu[model],
          });
        }
      }

      const gpu = setupJson.component_names?.gpu;
      if (gpu) {
        for (const model in gpu) {
          hwList.push({
            id: hwList.length + 1,
            type: "GPU",
            model: model,
            quantity: gpu[model],
          });
        }
      }
      console.log(hwList);
      setHardwareData(hwList);
    } catch (err) {
      console.error("Error fetching data:", err);
    }
  };

  return (
    <div className="enter-link-container">
      <h3 className="enter-link-heading">
        Enter the link to your RL pipeline:
      </h3>
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
      {hardwareData.length > 0 && (
        <HardwareSection initialHardware={hardwareData} />
      )}
    </div>
  );
}

export default EnterLink;
