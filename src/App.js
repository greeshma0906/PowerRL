import React from 'react';
import Header from './components/Header';
import EnterLink from './components/EnterLink';
import ModelEmission from './components/ModelEmission';
import PipelineTimeline from './components/PipelineTimeline';
import HardwareSection from './components/HardwareSection';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <EnterLink />
      <div className="content">
        <div className="left-panel">
          <ModelEmission />
        </div>
        <div className="right-panel">
          <PipelineTimeline />
          <HardwareSection />
        </div>
      </div>
    </div>
  );
}

export default App;
