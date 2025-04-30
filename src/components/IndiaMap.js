import React, { useEffect, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import indiaEnergyData from "../IndiaEnergyData.json";
import * as d3 from "d3";

const geoUrl = "/india-states.json"; // Should be placed in the public folder

const colorScale = d3
  .scaleLinear()
  .domain([0.0, 1.5])
  .range(["#b3cde0", "#003366"]);

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
      <h2 style={{ textAlign: "center" }}>CO₂ Intensity Across India</h2>

      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
          gap: "20px", // Reduced gap between map and bar
        }}
      >
        {/* Map Section */}
        <div style={{ flex: 1, display: "flex", flexDirection: "row", marginLeft: "200px" }}> {/* Added marginLeft */}
          <ComposableMap
            projection="geoMercator"
            projectionConfig={{ scale: 1000, center: [82.8, 22.5] }}
            style={{ width: "100%", height: "50vh" }}
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
                          ? "#66c2a5"
                          : value !== undefined
                          ? colorScale(value)
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
        </div>

        {/* Color Legend */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            height: "50vh", // Match the height of the map
            justifyContent: "center",
            marginTop: "10px", // Reduced top margin to reduce gap
          }}
        >
          <div
            style={{
              height: "100%", // Ensure the gradient covers the full height
              width: "20px",
              background: "linear-gradient(to bottom, #003366, #b3cde0)",
              border: "1px solid #ccc",
            }}
          />
          <div
            style={{
              marginTop: "10px",
              fontSize: "12px",
              textAlign: "center",
              marginBottom: "10px", // To keep space for the text
            }}
          >
            <strong>CO₂ (lb/kWh)</strong>
          </div>
          <div
            style={{
              fontSize: "12px",
              textAlign: "center",
              marginTop: "5px", // Reduced margin between bar and text
            }}
          >
            <small>Dark = High CO₂ intensity, Light = Low</small>
          </div>
        </div>
      </div>

      {/* Selected State Info */}
      {selectedState && selectedValue !== null && (
        <div
          style={{
            marginTop: 20,
            textAlign: "center",
            maxWidth: "400px",
            margin: "auto",
            padding: "20px",
            backgroundColor: "#fff",
            borderRadius: "8px",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
            border: "1px solid #ccc",
          }}
        >
          <h3 style={{ marginBottom: "10px", color: "#333" }}>
            State: {selectedState}
          </h3>
          <p style={{ marginBottom: "10px" }}>
            <strong>CO₂ Intensity:</strong> {selectedValue} lb/kWh
          </p>
          {averageEnergy !== undefined && (
            <>
              <p style={{ marginBottom: "10px" }}>
                <strong>Average Energy Used:</strong> {averageEnergy.toFixed(5)}{" "}
                kWh
              </p>
              <p>
                <strong>Estimated CO₂ Emissions:</strong>{" "}
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
