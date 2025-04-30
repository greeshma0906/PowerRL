// import React, { useState, useEffect } from "react";
// import "./HardwareSection.css";

// function HardwareSection({ initialHardware = [], averageEnergy }) {
//   const hardwareList = Array.isArray(initialHardware) ? initialHardware : [];

//   const [hardware, setHardware] = useState(
//     hardwareList.length > 0
//       ? hardwareList
//       : [
//           { id: 1, type: "CPU", model: "Intel i7", quantity: 1 },
//           { id: 2, type: "GPU", model: "GTX-1080-Ti", quantity: 1 },
//         ]
//   );

//   const [selectedHardware, setSelectedHardware] = useState("");
//   const [efficiencyData, setEfficiencyData] = useState({});
//   const [adjustedEnergy, setAdjustedEnergy] = useState(null);

//   const hardwareOptions = [
//     { value: "cpu-i9-12900k", label: "CPU - Intel i9 12900K" },
//     { value: "cpu-ryzen-9-5950x", label: "CPU - AMD Ryzen 9 5950X" },
//     { value: "gpu-rtx-3090", label: "GPU - NVIDIA RTX 3090" },
//     { value: "gpu-rtx-3080", label: "GPU - NVIDIA RTX 3080" },
//     { value: "gpu-radeon-rx-6900xt", label: "GPU - AMD Radeon RX 6900XT" },
//   ];

//   useEffect(() => {
//     fetch("/efficiency.json")
//       .then((res) => res.json())
//       .then((data) => setEfficiencyData(data))
//       .catch((err) => console.error("Failed to load efficiency data:", err));
//   }, []);

//   useEffect(() => {
//     if (averageEnergy && selectedHardware) {
//       const selected = hardwareOptions.find(
//         (opt) => opt.value === selectedHardware
//       );
//       const label = selected?.label;
//       const multiplier = efficiencyData[label] ?? 1;
//       setAdjustedEnergy((averageEnergy * multiplier).toFixed(5));
//     } else {
//       setAdjustedEnergy(null);
//     }
//   }, [averageEnergy, selectedHardware, efficiencyData]);

//   return (
//     <div className="hardware-section">
//       <h3>Your Hardware</h3>

//       <div className="hardware-list">
//         {hardware.map((item) => (
//           <div key={item.id} className="hardware-item">
//             <div className="hardware-icon">
//               {item.type === "CPU" ? "🔲" : "📊"}
//             </div>
//             <div className="hardware-details">
//               <div className="hardware-name">
//                 {item.type} - {item.model}
//               </div>
//               <div className="hardware-quantity">Quantity: {item.quantity}</div>
//             </div>
//           </div>
//         ))}
//       </div>

//       <hr />

//       <div className="add-hardware">
//         <h4>Select Hardware for Efficiency Calculation</h4>
//         <select
//           value={selectedHardware}
//           onChange={(e) => setSelectedHardware(e.target.value)}
//           className="hardware-select"
//         >
//           <option value="">Select Hardware</option>
//           {hardwareOptions.map((option) => (
//             <option key={option.value} value={option.value}>
//               {option.label}
//             </option>
//           ))}
//         </select>
//       </div>

//       {adjustedEnergy !== null && (
//         <div className="result-display">
//           <h4>Adjusted Energy Consumption</h4>
//           <p>
//             <strong>{adjustedEnergy} kWh</strong> (based on selected hardware
//             efficiency)
//           </p>
//         </div>
//       )}
//     </div>
//   );
// }

// export default HardwareSection;
import React, { useState, useEffect } from "react";
import "./HardwareSection.css";

function HardwareSection({ initialHardware = [], averageEnergy, setAdjustedEnergy }) {
  const hardwareList = Array.isArray(initialHardware) ? initialHardware : [];

  const [hardware, setHardware] = useState(
    hardwareList.length > 0
      ? hardwareList
      : [
          { id: 1, type: "CPU", model: "Intel i7", quantity: 1 },
          { id: 2, type: "GPU", model: "GTX-1080-Ti", quantity: 1 },
        ]
  );

  const [selectedHardware, setSelectedHardware] = useState("");
  const [efficiencyData, setEfficiencyData] = useState({});
  const [localAdjustedEnergy, setLocalAdjustedEnergy] = useState(null);

  const hardwareOptions = [
    { value: "cpu-i9-12900k", label: "CPU - Intel i9 12900K" },
    { value: "cpu-ryzen-9-5950x", label: "CPU - AMD Ryzen 9 5950X" },
    { value: "gpu-rtx-3090", label: "GPU - NVIDIA RTX 3090" },
    { value: "gpu-rtx-3080", label: "GPU - NVIDIA RTX 3080" },
    { value: "gpu-radeon-rx-6900xt", label: "GPU - AMD Radeon RX 6900XT" },
  ];

  useEffect(() => {
    fetch("/efficiency.json")
      .then((res) => res.json())
      .then((data) => setEfficiencyData(data))
      .catch((err) => console.error("Failed to load efficiency data:", err));
  }, []);

  useEffect(() => {
    if (averageEnergy && selectedHardware) {
      const selected = hardwareOptions.find(
        (opt) => opt.value === selectedHardware
      );
      const label = selected?.label;
      const multiplier = efficiencyData[label] ?? 1;
      const adjusted = (averageEnergy * multiplier).toFixed(5);
      setLocalAdjustedEnergy(adjusted);
      setAdjustedEnergy && setAdjustedEnergy(Number(adjusted)); // pass up to parent
    } else {
      setLocalAdjustedEnergy(null);
      setAdjustedEnergy && setAdjustedEnergy(null);
    }
  }, [averageEnergy, selectedHardware, efficiencyData]);

  return (
    <div className="hardware-section">
      <h3>Your Hardware</h3>

      <div className="hardware-list">
        {hardware.map((item) => (
          <div key={item.id} className="hardware-item">
            <div className="hardware-icon">
              {item.type === "CPU" ? "🔲" : "📊"}
            </div>
            <div className="hardware-details">
              <div className="hardware-name">
                {item.type} - {item.model}
              </div>
              <div className="hardware-quantity">Quantity: {item.quantity}</div>
            </div>
          </div>
        ))}
      </div>

      <hr />

      <div className="add-hardware">
        <h4>Select Hardware for Efficiency Calculation</h4>
        <select
          value={selectedHardware}
          onChange={(e) => setSelectedHardware(e.target.value)}
          className="hardware-select"
        >
          <option value="">Select Hardware</option>
          {hardwareOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {localAdjustedEnergy !== null && (
        <div className="result-display">
          <h4>Adjusted Energy Consumption</h4>
          <p>
            <strong>{localAdjustedEnergy} kWh</strong> (based on selected hardware
            efficiency)
          </p>
        </div>
      )}
    </div>
  );
}

export default HardwareSection;
