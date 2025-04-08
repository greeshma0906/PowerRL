// import React, { useState } from 'react';
// import './EnterLink.css';
// import ModelEmissions from './ModelEmission';
// import HardwareSection from './HardwareSection';

// function EnterLink() {
//   const [url, setUrl] = useState('');
//   const [chartData, setChartData] = useState(null);
//   const [hardwareData, setHardwareData] = useState([]);

//   // const handleFetchData = async () => {
//   //   try {
//   //     // Fetch emissions data
//   //     const energyRes = await fetch(`${url}/energy-stats`);
//   //     const energyJson = await energyRes.json();

//   //     const epochs = energyJson.map((entry, index) => index.toString());
//   //     const emissions = energyJson.map(entry => entry.energy);

//   //     const data = {
//   //       labels: epochs,
//   //       datasets: [
//   //         {
//   //           label: 'Carbon Emissions (CO2 lbs)',
//   //           data: emissions,
//   //           borderColor: 'rgba(75, 192, 75, 1)',
//   //           backgroundColor: 'rgba(75, 192, 75, 0.2)',
//   //           borderWidth: 2,
//   //           pointRadius: 0,
//   //           tension: 0.3,
//   //         }
//   //       ],
//   //     };
//   //     setChartData(data);

//   //     // Fetch hardware setup data
//   //     const setupRes = await fetch(`${url}/initial-setup`);
//   //     const setupJson = await setupRes.json();

//   //     const hwList = [];

//   //     const cpu = setupJson.component_names?.cpu;
//   //     if (cpu) {
//   //       for (const model in cpu) {
//   //         hwList.push({
//   //           id: hwList.length + 1,
//   //           type: 'CPU',
//   //           model: model,
//   //           quantity: cpu[model],
//   //         });
//   //       }
//   //     }

//   //     const gpu = setupJson.component_names?.gpu;
//   //     if (gpu) {
//   //       for (const model in gpu) {
//   //         hwList.push({
//   //           id: hwList.length + 1,
//   //           type: 'GPU',
//   //           model: model,
//   //           quantity: gpu[model],
//   //         });
//   //       }
//   //     }
//   //     console.log(hwList);
//   //     setHardwareData(hwList);

//   //   } catch (err) {
//   //     console.error('Error fetching data:', err);
//   //   }
//   // };
//   const handleFetchData = async () => {
//     try {
//       // Fetch emissions data
//       const energyRes = await fetch(`${url}/energy-stats`);
//       const energyJson = await energyRes.json();
  
//       const intervalEntries = Object.entries(energyJson.cpu.interval);
//       const epochs = intervalEntries.map(([key]) => key);       // keys are "0", "1", ...
//       const emissions = intervalEntries.map(([, value]) => value); // values are the emissions
  
//       const data = {
//         labels: epochs,
//         datasets: [
//           {
//             label: 'Carbon Emissions (CO2 lbs)',
//             data: emissions,
//             borderColor: 'rgba(75, 192, 75, 1)',
//             backgroundColor: 'rgba(75, 192, 75, 0.2)',
//             borderWidth: 2,
//             pointRadius: 0,
//             tension: 0.3,
//           }
//         ],
//       };
//       setChartData(data);
  
//       // Fetch hardware setup data
//       const setupRes = await fetch(`${url}/initial-setup`);
//       const setupJson = await setupRes.json();
  
//       const hwList = [];
  
//       const cpu = setupJson.component_names?.cpu;
//       if (cpu) {
//         for (const model in cpu) {
//           hwList.push({
//             id: hwList.length + 1,
//             type: 'CPU',
//             model: model,
//             quantity: cpu[model],
//           });
//         }
//       }
  
//       const gpu = setupJson.component_names?.gpu;
//       if (gpu) {
//         for (const model in gpu) {
//           hwList.push({
//             id: hwList.length + 1,
//             type: 'GPU',
//             model: model,
//             quantity: gpu[model],
//           });
//         }
//       }
  
//       console.log(hwList);
//       setHardwareData(hwList);
  
//     } catch (err) {
//       console.error('Error fetching data:', err);
//     }
//   };
  
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
  const [chartData, setChartData] = useState(null);
  const [hardwareData, setHardwareData] = useState([]);
  const [intervalId, setIntervalId] = useState(null);

  const fetchAndUpdateData = async () => {
    try {
      // Fetch emissions data
      const energyRes = await fetch(`${url}/energy-stats`);
      const energyJson = await energyRes.json();

      const intervalData = energyJson.cpu?.interval;
      if (intervalData) {
        const epochs = Object.keys(intervalData);
        const emissions = Object.values(intervalData);

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
            },
          ],
        };

        setChartData(data);
      }

      // Fetch hardware setup (optional: only fetch once if desired)
      const setupRes = await fetch(`${url}/initial-setup`);
      const setupJson = await setupRes.json();

      const hwList = [];
      const cpu = setupJson.component_names?.cpu;
      if (cpu && typeof cpu === 'object') {
        for (const model in cpu) {
          console.log(model)
          hwList.push({
            id: hwList.length + 1,
            type: 'CPU',
            model,
            quantity: cpu[model],
          });
        }
      }

      const gpu = setupJson.component_names?.gpu;
      if (gpu && typeof gpu === 'object' && Object.keys(gpu).length > 0) {
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

      {chartData && <ModelEmissions chartData={chartData} />}
      {hardwareData.length > 0 && <HardwareSection initialHardware={hardwareData} />}
    </div>
  );
}

export default EnterLink;
