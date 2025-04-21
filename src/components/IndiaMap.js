// import React, { useEffect, useState } from "react";
// import { ComposableMap, Geographies, Geography } from "react-simple-maps";
// import indiaEnergyData from "../IndiaEnergyData.json";
// import * as d3 from "d3";

// const geoUrl = "/india-states.json"; // from public folder

// const colorScale = d3
//   .scaleLinear()
//   .domain([0.0, 1.5]) // Adjust max value as needed
//   .range(["#b3cde0", "#003366"]); // Light blue to dark blue

// const IndiaMap = () => {
//   const [geoData, setGeoData] = useState([]);
//   const [selectedState, setSelectedState] = useState(null);
//   const [selectedValue, setSelectedValue] = useState(null);

//   useEffect(() => {
//     fetch(geoUrl)
//       .then((res) => res.json())
//       .then((data) => setGeoData(data.features));
//   }, []);

//   const handleStateClick = (stateName) => {
//     const value = indiaEnergyData[stateName];
//     setSelectedState(stateName);
//     setSelectedValue(value);
//   };

//   return (
//     <div style={{ padding: 20 }}>
//       <h2 style={{ textAlign: "center" }}>India Energy Intensity (CO₂/kWh)</h2>
//       <ComposableMap
//         projection="geoMercator"
//         projectionConfig={{ scale: 1000, center: [82.8, 22.5] }}
//         style={{ width: "100%", height: "auto" }}
//       >
//         <Geographies
//           geography={{ type: "FeatureCollection", features: geoData }}
//         >
//           {({ geographies }) =>
//             geographies.map((geo) => {
//               const stateName = geo.properties.st_nm;
//               const value = indiaEnergyData[stateName];
//               return (
//                 <Geography
//                   key={geo.rsmKey}
//                   geography={geo}
//                   fill={value !== undefined ? colorScale(value) : "#EEE"}
//                   stroke="#FFF"
//                   onClick={() => handleStateClick(stateName)} // Add click handler
//                   style={{
//                     default: { outline: "none" },
//                     hover: { fill: "#ffa500", outline: "none" },
//                     pressed: { outline: "none" },
//                   }}
//                 />
//               );
//             })
//           }
//         </Geographies>
//       </ComposableMap>
//       <div style={{ textAlign: "center", marginTop: 10 }}>
//         <small>Dark = High CO₂ intensity, Light = Low</small>
//       </div>

//       {/* Display the selected state's CO₂ intensity */}
//       {selectedState && selectedValue !== null && (
//         <div style={{ marginTop: 20, textAlign: "center" }}>
//           <h3>State: {selectedState}</h3>
//           <p>CO₂ Intensity: {selectedValue}  lb/kWh</p>
//         </div>
//       )}
//     </div>
//   );
// };

// export default IndiaMap;

import React, { useEffect, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import indiaEnergyData from "../IndiaEnergyData.json";
import * as d3 from "d3";

const geoUrl = "/india-states.json"; // from public folder

const colorScale = d3
  .scaleLinear()
  .domain([0.0, 1.5]) // Adjust max value as needed
  .range(["#b3cde0", "#003366"]); // Light blue to dark blue

const IndiaMap = ({ averageEnergy }) => {
  const [geoData, setGeoData] = useState([]);
  const [selectedState, setSelectedState] = useState(null);
  const [selectedValue, setSelectedValue] = useState(null);

  useEffect(() => {
    fetch(geoUrl)
      .then((res) => res.json())
      .then((data) => setGeoData(data.features));
  }, []);

  const handleStateClick = (stateName) => {
    const value = indiaEnergyData[stateName];
    setSelectedState(stateName);
    setSelectedValue(value);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2 style={{ textAlign: "center" }}>India Energy Intensity (CO₂/kWh)</h2>
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ scale: 1000, center: [82.8, 22.5] }}
        style={{ width: "100%", height: "auto" }}
      >
        <Geographies
          geography={{ type: "FeatureCollection", features: geoData }}
        >
          {({ geographies }) =>
            geographies.map((geo) => {
              const stateName = geo.properties.st_nm;
              const value = indiaEnergyData[stateName];
              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  onClick={() => handleStateClick(stateName)}
                  fill={
                    selectedState === stateName
                      ? "#66c2a5" // selected
                      : value !== undefined
                      ? colorScale(value) // default color
                      : "#EEE"
                  }
                  stroke="#FFF"
                  style={{
                    default: { outline: "none" },
                    hover: { fill: "#66c2a5", outline: "none" },
                    pressed: { fill: "#66c2a5", outline: "none" },
                  }}
                />
              );
            })
          }
        </Geographies>
      </ComposableMap>

      <div style={{ textAlign: "center", marginTop: 10 }}>
        <small>Dark = High CO₂ intensity, Light = Low</small>
      </div>

      {/* Display selected state's CO₂ intensity and total emissions */}
      {selectedState && selectedValue !== null && (
        <div style={{ marginTop: 20, textAlign: "center" }}>
          <h3>State: {selectedState}</h3>
          <p>CO₂ Intensity: {selectedValue} lb/kWh</p>
          {averageEnergy !== undefined && (
            <>
              <p>Average Energy Used: {averageEnergy.toFixed(5)} kWh</p>
              <p>
                Estimated CO₂ Emissions:{" "}
                {(averageEnergy * selectedValue).toFixed(5)} lbs
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default IndiaMap;
