// // // import React, { useState, useEffect } from 'react';
// // // import './EnterLink.css';
// // // import ModelEmissions from './ModelEmission';
// // // import HardwareSection from './HardwareSection';

// // // function EnterLink() {
// // //   const [url, setUrl] = useState('');
// // //   const [chartData, setChartData] = useState(null);
// // //   const [hardwareData, setHardwareData] = useState([]);
// // //   const [intervalId, setIntervalId] = useState(null);

// // //   const fetchAndUpdateData = async () => {
// // //     try {
// // //       // Fetch emissions data
// // //       const energyRes = await fetch(`${url}/energy-stats`);
// // //       const energyJson = await energyRes.json();

// // //       const intervalData = energyJson.cpu?.interval;
// // //       if (intervalData) {
// // //         const epochs = Object.keys(intervalData);
// // //         const emissions = Object.values(intervalData);

// // //         const data = {
// // //           labels: epochs,
// // //           datasets: [
// // //             {
// // //               label: 'Carbon Emissions (CO2 lbs)',
// // //               data: emissions,
// // //               borderColor: 'rgba(75, 192, 75, 1)',
// // //               backgroundColor: 'rgba(75, 192, 75, 0.2)',
// // //               borderWidth: 2,
// // //               pointRadius: 0,
// // //               tension: 0.3,
// // //             },
// // //           ],
// // //         };

// // //         setChartData(data);
// // //       }

// // //       // Fetch hardware setup (optional: only fetch once if desired)
// // //       const setupRes = await fetch(`${url}/initial-setup`);
// // //       const setupJson = await setupRes.json();

// // //       const hwList = [];
// // //       const cpu = setupJson.component_names?.cpu;
// // //       if (cpu && typeof cpu === 'object') {
// // //         for (const model in cpu) {
// // //           console.log(model)
// // //           hwList.push({
// // //             id: hwList.length + 1,
// // //             type: 'CPU',
// // //             model,
// // //             quantity: cpu[model],
// // //           });
// // //         }
// // //       }

// // //       const gpu = setupJson.component_names?.gpu;
// // //       if (gpu && typeof gpu === 'object' && Object.keys(gpu).length > 0) {
// // //         for (const model in gpu) {
// // //           hwList.push({
// // //             id: hwList.length + 1,
// // //             type: 'GPU',
// // //             model,
// // //             quantity: gpu[model],
// // //           });
// // //         }
// // //       }

// // //       setHardwareData(hwList);
// // //     } catch (err) {
// // //       console.error('Error fetching data:', err);
// // //     }
// // //   };

// // //   const handleFetchData = async () => {
// // //     if (intervalId) {
// // //       clearInterval(intervalId);
// // //     }

// // //     // Initial fetch
// // //     await fetchAndUpdateData();

// // //     // Periodic fetching every 5 seconds
// // //     const newIntervalId = setInterval(fetchAndUpdateData, 5000);
// // //     setIntervalId(newIntervalId);
// // //   };

// // //   // Cleanup on unmount
// // //   useEffect(() => {
// // //     return () => {
// // //       if (intervalId) {
// // //         clearInterval(intervalId);
// // //       }
// // //     };
// // //   }, [intervalId]);

// // //   return (
// // //     <div className="enter-link-container">
// // //       <h3 className="enter-link-heading">Enter the link to your RL pipeline:</h3>
// // //       <div className="enter-link-bar">
// // //         <input
// // //           type="text"
// // //           placeholder="Paste your link here..."
// // //           className="link-input"
// // //           value={url}
// // //           onChange={(e) => setUrl(e.target.value)}
// // //         />
// // //         <button className="measure-button" onClick={handleFetchData}>
// // //           Measure
// // //         </button>
// // //       </div>

// // //       {chartData && <ModelEmissions chartData={chartData} />}
// // //       {hardwareData.length > 0 && <HardwareSection initialHardware={hardwareData} />}
// // //     </div>
// // //   );
// // // }

// // // export default EnterLink;
// // import React, { useState, useEffect } from 'react';
// // import './EnterLink.css';
// // import ModelEmissions from './ModelEmission';
// // import HardwareSection from './HardwareSection';

// // function EnterLink() {
// //   const [url, setUrl] = useState('');
// //   const [chartData, setChartData] = useState(null);
// //   const [hardwareData, setHardwareData] = useState([]);
// //   const [intervalId, setIntervalId] = useState(null);

// //   const fetchAndUpdateData = async () => {
// //     try {
// //       // Fetch emissions data
// //       const energyRes = await fetch(`${url}/energy-stats`);
// //       const energyJson = await energyRes.json();

// //       const cpuData = energyJson.cpu?.interval || {};
// //       const gpuData = energyJson.gpu?.interval || {};

// //       // Use CPU keys as the timeline reference
// //       const epochs = Object.keys(cpuData);

// //       const cpuEmissions = epochs.map(epoch => cpuData[epoch] || 0);
// //       const gpuEmissions = epochs.map(epoch => gpuData[epoch] || 0);

// //       const data = {
// //         labels: epochs,
// //         datasets: [
// //           {
// //             label: 'CPU Emissions (CO2 lbs)',
// //             data: cpuEmissions,
// //             borderColor: 'rgba(75, 192, 75, 1)',
// //             backgroundColor: 'rgba(75, 192, 75, 0.2)',
// //             borderWidth: 2,
// //             pointRadius: 0,
// //             tension: 0.3,
// //           },
// //           {
// //             label: 'GPU Emissions (CO2 lbs)',
// //             data: gpuEmissions,
// //             borderColor: 'rgba(255, 99, 132, 1)',
// //             backgroundColor: 'rgba(255, 99, 132, 0.2)',
// //             borderWidth: 2,
// //             pointRadius: 0,
// //             tension: 0.3,
// //           },
// //         ],
// //       };

// //       setChartData(data);

// //       // Fetch hardware setup
// //       const setupRes = await fetch(`${url}/initial-setup`);
// //       const setupJson = await setupRes.json();

// //       const hwList = [];

// //       const cpu = setupJson.component_names?.cpu;
// //       if (cpu && typeof cpu === 'object') {
// //         for (const model in cpu) {
// //           hwList.push({
// //             id: hwList.length + 1,
// //             type: 'CPU',
// //             model,
// //             quantity: cpu[model],
// //           });
// //         }
// //       }

// //       const gpu = setupJson.component_names?.gpu;
// //       if (gpu && typeof gpu === 'object') {
// //         for (const model in gpu) {
// //           hwList.push({
// //             id: hwList.length + 1,
// //             type: 'GPU',
// //             model,
// //             quantity: gpu[model],
// //           });
// //         }
// //       }

// //       setHardwareData(hwList);
// //     } catch (err) {
// //       console.error('Error fetching data:', err);
// //     }
// //   };

// //   const handleFetchData = async () => {
// //     if (intervalId) {
// //       clearInterval(intervalId);
// //     }

// //     // Initial fetch
// //     await fetchAndUpdateData();

// //     // Periodic fetching every 5 seconds
// //     const newIntervalId = setInterval(fetchAndUpdateData, 5000);
// //     setIntervalId(newIntervalId);
// //   };

// //   // Cleanup on unmount
// //   useEffect(() => {
// //     return () => {
// //       if (intervalId) {
// //         clearInterval(intervalId);
// //       }
// //     };
// //   }, [intervalId]);

// //   return (
// //     <div className="enter-link-container">
// //       <h3 className="enter-link-heading">Enter the link to your RL pipeline:</h3>
// //       <div className="enter-link-bar">
// //         <input
// //           type="text"
// //           placeholder="Paste your link here..."
// //           className="link-input"
// //           value={url}
// //           onChange={(e) => setUrl(e.target.value)}
// //         />
// //         <button className="measure-button" onClick={handleFetchData}>
// //           Measure
// //         </button>
// //       </div>

// //       {chartData && <ModelEmissions chartData={chartData} />}
// //       {hardwareData.length > 0 && <HardwareSection initialHardware={hardwareData} />}
// //     </div>
// //   );
// // }

// // export default EnterLink;
// import React, { useState, useEffect } from 'react';
// import './EnterLink.css';
// import ModelEmissions from './ModelEmission';
// import HardwareSection from './HardwareSection';

// function EnterLink() {
//   const [url, setUrl] = useState('');
//   const [chartData, setChartData] = useState(null);
//   const [hardwareData, setHardwareData] = useState([]);
//   const [intervalId, setIntervalId] = useState(null);

//   const fetchAndUpdateData = async () => {
//     try {
//       const energyRes = await fetch(`${url}/energy-stats`);
//       const energyJson = await energyRes.json();

//       const cpuInterval = energyJson.cpu?.interval || {};
//       const gpuInterval = energyJson.gpu?.interval || {};
//       const cpuEpoch = energyJson.cpu?.epoch || {};
//       const gpuEpoch = energyJson.gpu?.epoch || {};

//       const epochs = Object.keys(cpuInterval);

//       const cpuIntervalEmissions = epochs.map(epoch => cpuInterval[epoch] || 0);
//       const gpuIntervalEmissions = epochs.map(epoch => gpuInterval[epoch] || 0);
//       const cpuEpochEmissions = epochs.map(epoch => cpuEpoch[epoch] ?? null);
//       const gpuEpochEmissions = epochs.map(epoch => gpuEpoch[epoch] ?? null);

//       const data = {
//         labels: epochs,
//         datasets: [
//           {
//             label: 'CPU Interval Emissions (CO2 lbs)',
//             data: cpuIntervalEmissions,
//             borderColor: 'rgba(75, 192, 75, 1)',
//             backgroundColor: 'rgba(75, 192, 75, 0.2)',
//             borderWidth: 2,
//             pointRadius: 0,
//             tension: 0.3,
//           },
//           {
//             label: 'GPU Interval Emissions (CO2 lbs)',
//             data: gpuIntervalEmissions,
//             borderColor: 'rgba(255, 99, 132, 1)',
//             backgroundColor: 'rgba(255, 99, 132, 0.2)',
//             borderWidth: 2,
//             pointRadius: 0,
//             tension: 0.3,
//           },
//           {
//             label: 'CPU Epoch Emissions (CO2 lbs)',
//             data: cpuEpochEmissions,
//             borderColor: 'rgba(54, 162, 235, 1)',
//             backgroundColor: 'rgba(54, 162, 235, 0.2)',
//             borderDash: [5, 5],
//             borderWidth: 2,
//             pointRadius: 2,
//             tension: 0.3,
//           },
//           {
//             label: 'GPU Epoch Emissions (CO2 lbs)',
//             data: gpuEpochEmissions,
//             borderColor: 'rgba(255, 206, 86, 1)',
//             backgroundColor: 'rgba(255, 206, 86, 0.2)',
//             borderDash: [5, 5],
//             borderWidth: 2,
//             pointRadius: 2,
//             tension: 0.3,
//           },
//         ],
//       };

//       setChartData(data);

//       // Fetch hardware setup
//       const setupRes = await fetch(`${url}/initial-setup`);
//       const setupJson = await setupRes.json();

//       const hwList = [];

//       const cpu = setupJson.component_names?.cpu;
//       if (cpu && typeof cpu === 'object') {
//         for (const model in cpu) {
//           hwList.push({
//             id: hwList.length + 1,
//             type: 'CPU',
//             model,
//             quantity: cpu[model],
//           });
//         }
//       }

//       const gpu = setupJson.component_names?.gpu;
//       if (gpu && typeof gpu === 'object') {
//         for (const model in gpu) {
//           hwList.push({
//             id: hwList.length + 1,
//             type: 'GPU',
//             model,
//             quantity: gpu[model],
//           });
//         }
//       }

//       setHardwareData(hwList);
//     } catch (err) {
//       console.error('Error fetching data:', err);
//     }
//   };

//   const handleFetchData = async () => {
//     if (intervalId) {
//       clearInterval(intervalId);
//     }

//     // Initial fetch
//     await fetchAndUpdateData();

//     // Periodic fetching every 5 seconds
//     const newIntervalId = setInterval(fetchAndUpdateData, 5000);
//     setIntervalId(newIntervalId);
//   };

//   // Cleanup on unmount
//   useEffect(() => {
//     return () => {
//       if (intervalId) {
//         clearInterval(intervalId);
//       }
//     };
//   }, [intervalId]);

//   return (
//     <div className="enter-link-container">
//       <h3 className="enter-link-heading">Enter the link to your RL pipeline:</h3>
//       <div className="enter-link-bar">
//         <input
//           type="text"
//           placeholder="Paste your link here..."
//           className="link-input"
//           value={url}
//           onChange={(e) => setUrl(e.target.value)}
//         />
//         <button className="measure-button" onClick={handleFetchData}>
//           Measure
//         </button>
//       </div>

//       {chartData && <ModelEmissions chartData={chartData} />}
//       {hardwareData.length > 0 && <HardwareSection initialHardware={hardwareData} />}
//     </div>
//   );
// }

// export default EnterLink;
import React, { useState, useEffect } from 'react';
import './EnterLink.css';
import ModelEmissions from './ModelEmission';
import HardwareSection from './HardwareSection';

function EnterLink() {
  const [url, setUrl] = useState('');
  const [intervalData, setIntervalData] = useState(null);
  const [epochData, setEpochData] = useState(null);
  const [hardwareData, setHardwareData] = useState([]);
  const [intervalId, setIntervalId] = useState(null);

  const fetchAndUpdateData = async () => {
    try {
      const energyRes = await fetch(`${url}/energy-stats`);
      const energyJson = await energyRes.json();

      const cpuInterval = energyJson.cpu?.interval || {};
      const gpuInterval = energyJson.gpu?.interval || {};
      const cpuEpoch = energyJson.cpu?.epoch || {};
      const gpuEpoch = energyJson.gpu?.epoch || {};

      const epochs = Object.keys(cpuInterval);

      const cpuIntervalEmissions = epochs.map(epoch => cpuInterval[epoch] || 0);
      const gpuIntervalEmissions = epochs.map(epoch => gpuInterval[epoch] || 0);
      const cpuEpochEmissions = epochs.map(epoch => cpuEpoch[epoch] ?? null);
      const gpuEpochEmissions = epochs.map(epoch => gpuEpoch[epoch] ?? null);

      // Set interval data (for graph 1)
      setIntervalData({
        labels: epochs,
        datasets: [
          {
            label: 'CPU Interval Emissions (kWH)',
            data: cpuIntervalEmissions,
            borderColor: 'rgba(75, 192, 75, 1)',
            backgroundColor: 'rgba(75, 192, 75, 0.2)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
          },
          {
            label: 'GPU Interval Emissions (kWH)',
            data: gpuIntervalEmissions,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
          },
        ],
      });

      // Set epoch data (for graph 2)
      setEpochData({
        labels: epochs,
        datasets: [
          {
            label: 'CPU Epoch Emissions (kWH)',
            data: cpuEpochEmissions,
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
          },
          {
            label: 'GPU Epoch Emissions (kWH)',
            data: gpuEpochEmissions,
            borderColor: 'rgba(255, 206, 86, 1)',
            backgroundColor: 'rgba(255, 206, 86, 0.2)',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
          },
        ],
      });

      // Fetch hardware setup
      const setupRes = await fetch(`${url}/initial-setup`);
      const setupJson = await setupRes.json();

      const hwList = [];
      const cpu = setupJson.component_names?.cpu;
      if (cpu && typeof cpu === 'object') {
        for (const model in cpu) {
          hwList.push({
            id: hwList.length + 1,
            type: 'CPU',
            model,
            quantity: cpu[model],
          });
        }
      }

      const gpu = setupJson.component_names?.gpu;
      if (gpu && typeof gpu === 'object') {
        for (const model in gpu) {
          hwList.push({
            id: hwList.length + 1,
            type: 'GPU',
            model,
            quantity: gpu[model],
          });
        }
      }

      setHardwareData(hwList);
    } catch (err) {
      console.error('Error fetching data:', err);
    }
  };

  const handleFetchData = async () => {
    if (intervalId) {
      clearInterval(intervalId);
    }

    // Initial fetch
    await fetchAndUpdateData();

    // Periodic fetching every 5 seconds
    const newIntervalId = setInterval(fetchAndUpdateData, 5000);
    setIntervalId(newIntervalId);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);

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

      {intervalData && <ModelEmissions chartData={intervalData} title="Interval Emissions (CO2 lbs)" isEpochData={false} />}
      {epochData && <ModelEmissions chartData={epochData} title="Epoch Emissions (CO2 lbs)" isEpochData={true} />}

      {hardwareData.length > 0 && <HardwareSection initialHardware={hardwareData} />}
    </div>
  );
}

export default EnterLink;


