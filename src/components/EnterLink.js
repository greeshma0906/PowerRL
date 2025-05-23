import React, { useState, useEffect } from "react";
import "./EnterLink.css";
import ModelEmissions from "./ModelEmission";
import HardwareSection from "./HardwareSection";
import IndiaMap from "./IndiaMap";

function EnterLink() {
  const [rlUrl, setRlUrl] = useState("");
  const [nonRlUrl, setNonRlUrl] = useState("");

  const [rlIntervalData, setRlIntervalData] = useState(null);
  const [rlEpochData, setRlEpochData] = useState(null);
  const [nonRlIntervalData, setNonRlIntervalData] = useState(null);
  const [nonRlEpochData, setNonRlEpochData] = useState(null);
  const [hardwareData, setHardwareData] = useState([]);
  const [intervalId, setIntervalId] = useState(null);
  const [averageEnergy, setAverageEnergy] = useState(null);
  const [adjustedEnergy, setAdjustedEnergy] = useState(null);
  const [rlbleuData,setBleuDataRl] = useState(null);
  const [rlbleuDataNonRl,setBleuDataNonRl] = useState(null);

  // const fetchEnergyData = async (url, setIntervalData, setEpochData) => {
  //   try {
  //     const energyRes = await fetch(`${url}/energy-stats`);
  //     const energyJson = await energyRes.json();

  //     const cpuInterval = energyJson.cpu?.interval || {};
  //     const gpuInterval = energyJson.gpu?.interval || {};
  //     const cpuEpoch = energyJson.cpu?.epoch || {};
  //     const gpuEpoch = energyJson.gpu?.epoch || {};

  //     const epochs = Object.keys(cpuInterval);

  //     const cpuIntervalEmissions = epochs.map(
  //       (epoch) => cpuInterval[epoch] || 0
  //     );
  //     const gpuIntervalEmissions = epochs.map(
  //       (epoch) => gpuInterval[epoch] || 0
  //     );
  //     const cpuEpochEmissions = epochs.map((epoch) => cpuEpoch[epoch] ?? null);
  //     const gpuEpochEmissions = epochs.map((epoch) => gpuEpoch[epoch] ?? null);
  //     const totalEnergyPerEpoch = epochs.map((epoch) => {
  //       const cpu = cpuEpoch[epoch] ?? 0;
  //       const gpu = gpuEpoch[epoch] ?? 0;
  //       return cpu + gpu;
  //     });

  //     const validEnergies = totalEnergyPerEpoch.filter(
  //       (v) => v !== null && !isNaN(v)
  //     );
  //     const totalEnergyUsed = validEnergies.reduce((a, b) => a + b, 0);
  //     const avgEnergy =
  //       validEnergies.length > 0 ? totalEnergyUsed / validEnergies.length : 0;

  //     setAverageEnergy(Number(avgEnergy.toFixed(5)));

  //     setIntervalData({
  //       labels: epochs,
  //       datasets: [
  //         {
  //           label: "CPU Interval Emissions (kWH)",
  //           data: cpuIntervalEmissions,
  //           borderColor: "rgba(75, 192, 75, 1)",
  //           backgroundColor: "rgba(75, 192, 75, 0.2)",
  //           borderWidth: 2,
  //           pointRadius: 0,
  //           tension: 0.3,
  //         },
  //         {
  //           label: "GPU Interval Emissions (kWH)",
  //           data: gpuIntervalEmissions,
  //           borderColor: "rgba(255, 99, 132, 1)",
  //           backgroundColor: "rgba(255, 99, 132, 0.2)",
  //           borderWidth: 2,
  //           pointRadius: 0,
  //           tension: 0.3,
  //         },
  //       ],
  //     });

  //     setEpochData({
  //       labels: epochs,
  //       datasets: [
  //         {
  //           label: "CPU Epoch Emissions (kWH)",
  //           data: cpuEpochEmissions,
  //           borderColor: "rgba(54, 162, 235, 1)",
  //           backgroundColor: "rgba(54, 162, 235, 0.2)",
  //           borderDash: [5, 5],
  //           borderWidth: 2,
  //           pointRadius: 2,
  //           tension: 0.3,
  //         },
  //         {
  //           label: "GPU Epoch Emissions (kWH)",
  //           data: gpuEpochEmissions,
  //           borderColor: "rgba(255, 206, 86, 1)",
  //           backgroundColor: "rgba(255, 206, 86, 0.2)",
  //           borderDash: [5, 5],
  //           borderWidth: 2,
  //           pointRadius: 2,
  //           tension: 0.3,
  //         },
  //       ],
  //     });

  //     const setupRes = await fetch(`${url}/initial-setup`);
  //     const setupJson = await setupRes.json();

  //     const hwList = [];
  //     const cpu = setupJson.component_names?.cpu;
  //     if (cpu && typeof cpu === "object") {
  //       for (const model in cpu) {
  //         hwList.push({
  //           id: hwList.length + 1,
  //           type: "CPU",
  //           model,
  //           quantity: cpu[model],
  //         });
  //       }
  //     }

  //     const gpu = setupJson.component_names?.gpu;
  //     if (gpu && typeof gpu === "object") {
  //       for (const model in gpu) {
  //         hwList.push({
  //           id: hwList.length + 1,
  //           type: "GPU",
  //           model,
  //           quantity: gpu[model],
  //         });
  //       }
  //     }

  //     setHardwareData(hwList);
  //   } catch (err) {
  //     console.error("Error fetching data:", err);
  //   }
  // };
  
  const fetchEnergyData = async (url, setIntervalData, setEpochData,setBleuData) => {
    try {
      const energyRes = await fetch(`${url}/energy-stats`);
      const energyJson = await energyRes.json();
  
      const cpuInterval = energyJson.cpu?.interval || {};
      const gpuInterval = energyJson.gpu?.interval || {};
      const cpuEpoch = energyJson.cpu?.epoch || {};
      const gpuEpoch = energyJson.gpu?.epoch || {};
  
      const epochs = Object.keys(cpuInterval);
  
      const cpuIntervalEmissions = epochs.map((epoch) => cpuInterval[epoch] || 0);
      const gpuIntervalEmissions = epochs.map((epoch) => gpuInterval[epoch] || 0);
      const cpuEpochEmissions = epochs.map((epoch) => cpuEpoch[epoch] ?? null);
      const gpuEpochEmissions = epochs.map((epoch) => gpuEpoch[epoch] ?? null);
      const totalEnergyPerEpoch = epochs.map((epoch) => {
        const cpu = cpuEpoch[epoch] ?? 0;
        const gpu = gpuEpoch[epoch] ?? 0;
        return cpu + gpu;
      });
  
      const validEnergies = totalEnergyPerEpoch.filter((v) => v !== null && !isNaN(v));
      const totalEnergyUsed = validEnergies.reduce((a, b) => a + b, 0);
      const avgEnergy =
        validEnergies.length > 0 ? totalEnergyUsed / validEnergies.length : 0;
  
      setAverageEnergy(Number(avgEnergy.toFixed(5)));
  
      setIntervalData({
        labels: epochs,
        datasets: [
          {
            label: "CPU Interval Emissions (kWH)",
            data: cpuIntervalEmissions,
            borderColor: "rgba(75, 192, 75, 1)",
            backgroundColor: "rgba(75, 192, 75, 0.2)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
          },
          {
            label: "GPU Interval Emissions (kWH)",
            data: gpuIntervalEmissions,
            borderColor: "rgba(255, 99, 132, 1)",
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
          },
         
        ].filter(Boolean), // remove undefined if adjustedEnergy is null
      });
  
      setEpochData({
        labels: epochs,
        datasets: [
          {
            label: "CPU Epoch Emissions (kWH)",
            data: cpuEpochEmissions,
            borderColor: "rgba(54, 162, 235, 1)",
            backgroundColor: "rgba(54, 162, 235, 0.2)",
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
          },
          {
            label: "GPU Epoch Emissions (kWH)",
            data: gpuEpochEmissions,
            borderColor: "rgba(255, 206, 86, 1)",
            backgroundColor: "rgba(255, 206, 86, 0.2)",
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
          },
          setAdjustedEnergy && {
            label: "Adjusted Consumption (kWH)",
            data: new Array(epochs.length).fill(adjustedEnergy),
            borderColor: "rgb(141, 37, 115)",
            backgroundColor: "rgba(0, 123, 255, 0.1)",
            borderDash: [3, 3],
            borderWidth: 2,
            pointRadius: 1,
            tension: 0.3,
          },
        ].filter(Boolean),
      });
  
      const setupRes = await fetch(`${url}/initial-setup`);
      const setupJson = await setupRes.json();
  
      const hwList = [];
      const cpu = setupJson.component_names?.cpu;
      if (cpu && typeof cpu === "object") {
        for (const model in cpu) {
          hwList.push({
            id: hwList.length + 1,
            type: "CPU",
            model,
            quantity: cpu[model],
          });
        }
      }
  
      const gpu = setupJson.component_names?.gpu;
      if (gpu && typeof gpu === "object") {
        for (const model in gpu) {
          hwList.push({
            id: hwList.length + 1,
            type: "GPU",
            model,
            quantity: gpu[model],
          });
        }
      }
  
      setHardwareData(hwList);

    } catch (err) {
      console.error("Error fetching data:", err);
    }
  };

  const fetchBleuData = async (rlUrl, nonRlUrl, setBleuChartData) => {
    try {
      // Fetch RL BLEU scores
      const rlRes = await fetch(`${rlUrl}/get_bleu_rl`);
      const rlJson = await rlRes.json();
      const rlScores = (rlJson.logs || []).map(score => score * 10000);
      console.log(rlScores)
  
      // Fetch Non-RL BLEU scores
      const nonRlRes = await fetch(`${nonRlUrl}/get_bleu_rl`);
      const nonRlJson = await nonRlRes.json();
      const nonRlScores = nonRlJson.logs || [];
  
      // Create common labels
      const maxSteps = Math.max(rlScores.length, nonRlScores.length);
      const labels = Array.from({ length: maxSteps }, (_, i) => `Step ${i + 1}`);
  
      setBleuChartData({
        labels,
        datasets: [
          {
            label: "RL BLEU Score",
            data: rlScores,
            borderColor: "rgba(75, 192, 192, 1)",
            backgroundColor: "rgba(75, 192, 192, 0.2)",
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
          },
          {
            label: "Non-RL BLEU Score",
            data: nonRlScores,
            borderColor: "rgba(255, 159, 64, 1)",
            backgroundColor: "rgba(255, 159, 64, 0.2)",
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
          },
        ],
      });
    } catch (err) {
      console.error("Error fetching BLEU data:", err);
    }
  };
  
  
  
  const fetchAndUpdateData = async () => {
    try {
      await fetchEnergyData(rlUrl, setRlIntervalData, setRlEpochData,setBleuDataRl);
      await fetchEnergyData(nonRlUrl, setNonRlIntervalData, setNonRlEpochData,setBleuDataNonRl);
      await fetchBleuData(rlUrl,nonRlUrl, setBleuDataRl);
    } catch (error) {
      console.error("Error fetching energy data:", error);
    }
  };

  const handleFetchData = async () => {
    if (intervalId) {
      clearInterval(intervalId);
    }

    await fetchAndUpdateData();

    const newIntervalId = setInterval(fetchAndUpdateData, 5000);
    setIntervalId(newIntervalId);
  };

  const handleStopMonitoring = () => {
    if (intervalId) {
      clearInterval(intervalId);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);

  return (
    <div className="enter-link-wrapper">
      <div className="input-section">
        <div className="input-box">
          <h3>RL Pipeline Link:</h3>
          <input
            type="text"
            value={rlUrl}
            onChange={(e) => setRlUrl(e.target.value)}
            placeholder="Enter RL pipeline URL..."
          />
        </div>
        <div className="input-box">
          <h3>Non-RL Pipeline Link:</h3>
          <input
            type="text"
            value={nonRlUrl}
            onChange={(e) => setNonRlUrl(e.target.value)}
            placeholder="Enter Non-RL pipeline URL..."
          />
        </div>
        <div className="button-group">
          <button className="measure-button" onClick={handleFetchData}>
            Start Measuring
          </button>
        </div>
      </div>

      <div className="graph-section">
        {/* RL Section */}
        <div className="graph-container">
          <h3>RL Interval Emissions (kWH)</h3>
          <div className="graph-box">
            {rlIntervalData && (
              <ModelEmissions
                chartData={rlIntervalData}
                title="RL Interval Emissions (kWH)"
                isEpochData={false}
              />
            )}
          </div>
          <h3>RL Epoch Emissions (kWH)</h3>
          <div className="graph-box">
            {rlEpochData && (
              <ModelEmissions
                chartData={rlEpochData}
                title="RL Epoch Emissions (kWH)"
                isEpochData={true}
              />
            )}  
          </div>
         
        </div>

        <div className="vertical-line"></div>

        {/* Non-RL Section */}
        <div className="graph-container">
          <h3>Non-RL Interval Emissions (kWH)</h3>
          <div className="graph-box">
            {nonRlIntervalData && (
              <ModelEmissions
                chartData={nonRlIntervalData}
                title="Non-RL Interval Emissions (kWH)"
                isEpochData={false}
              />
            )}
          </div>
          <h3>Non-RL Epoch Emissions (kWH)</h3>
          <div className="graph-box">
            {nonRlEpochData && (
              <ModelEmissions
                chartData={nonRlEpochData}
                title="Non-RL Epoch Emissions (kWH)"
                isEpochData={true}
              />
            )}
          </div>
        </div>
        </div>
        <div className="hardware-bleu-wrapper">
  {hardwareData.length > 0 && (
    <div className="hardware-section">
     
      <HardwareSection
        initialHardware={hardwareData}
        averageEnergy={averageEnergy}
        setAdjustedEnergy={setAdjustedEnergy}
      />
    </div>
  )}

  <div className="bleu-graph-box">
    {rlbleuData && (
      <ModelEmissions
        chartData={rlbleuData}
        title="Bleu Score" 
        isEpochData={false}
      />
    )}
  </div>
</div>



      <div className="map-section">
        <IndiaMap averageEnergy={averageEnergy} />
      </div>
    </div>
  );
}

export default EnterLink;
