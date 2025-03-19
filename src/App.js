import React from "react";
import ChartComponent from "./components/ChartComponent";
import HardwareList from "./components/HardwareList";
import "./index.css"; // Ensure this file exists

function App() {
  return (
    <div className="container">
      <div className="box">
        <h2>Model Training Performance</h2>
        <ChartComponent />
      </div>
      <div className="box">
        <h2>Your Hardware</h2>
        <HardwareList />
      </div>
    </div>
  );
}

export default App;
