import React, { useState } from "react";
import { Plus, Minus } from "lucide-react";

const hardwareData = [
  { type: "CPU", name: "Intel i7 12600K", alternatives: 2 },
  { type: "GPU", name: "NVIDIA RTX 3080", alternatives: 5 },
  { type: "RAM", name: "32GB DDR4", alternatives: 3 },
];

function HardwareList() {
  const [hardware, setHardware] = useState(hardwareData);

  const handleChange = (index, value) => {
    setHardware((prev) =>
      prev.map((item, i) =>
        i === index ? { ...item, alternatives: Math.max(0, item.alternatives + value) } : item
      )
    );
  };

  return (
    <div>
      {hardware.map((item, index) => (
        <div key={index} className="hardware-item">
          <p>
            <strong>{item.type} - {item.name}</strong>
          </p>
          <p>Alternative: {item.alternatives}</p>
          <button onClick={() => handleChange(index, -1)}><Minus size={16} /></button>
          <button onClick={() => handleChange(index, 1)}><Plus size={16} /></button>
        </div>
      ))}
    </div>
  );
}

export default HardwareList;
