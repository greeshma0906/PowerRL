import React, { useState } from "react";
import ChartComponent from "./components/ChartComponent";
import HardwareList from "./components/HardwareList";
import "./index.css";

function App() {
  const [resetKey, setResetKey] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showOptions, setShowOptions] = useState(false);

  const handleReset = () => {
    setResetKey((prevKey) => prevKey + 1); // Triggers re-render of HardwareList
  };

  const handleTransform = () => {
    console.log("Transformer function triggered!");
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(`📂 ${file.name} (Computer)`);
    }
  };

  const handleDriveUpload = () => {
    // Mock function for Drive upload (Replace with real API logic)
    const driveFile = "example_from_drive.pdf";
    setUploadedFile(`📁 ${driveFile} (Drive)`);
  };

  return (
    <div className="container">
      {/* Top Buttons */}
      <div className="top-buttons">
        <button className="reset-btn" onClick={handleReset} aria-label="Reset">
          🔄 Reset
        </button>
        <button className="transform-btn" onClick={handleTransform} aria-label="Transform">
          ⚙️ Transform
        </button>

        {/* Single Import Button with Dropdown Options */}
        <div className="import-dropdown">
          <button
            className="import-btn"
            onClick={() => setShowOptions(!showOptions)}
            aria-label="Import"
          >
            📥 Import ▼
          </button>
          {showOptions && (
            <div className="import-options">
              <label className="import-option">
                📂 Upload from Computer
                <input type="file" onChange={handleFileUpload} hidden />
              </label>
              <button className="import-option" onClick={handleDriveUpload}>
                ☁️ Upload from Drive
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Display Uploaded File Name */}
      {uploadedFile && <p className="file-name">{uploadedFile}</p>}

      {/* Two Sections - Side by Side */}
      <div className="content">
        <div className="box">
          <h2>📊 Model Training Performance</h2>
          <ChartComponent />
        </div>
        <div className="box">
          <h2>🖥️ Your Hardware</h2>
          <HardwareList key={resetKey} />
        </div>
      </div>
    </div>
  );
}

export default App;
