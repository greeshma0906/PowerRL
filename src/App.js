// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;
import React from 'react';
import Header from './components/Header';
import EnterLink from './components/EnterLink';
import PipelineTimeline from './components/PipelineTimeline'; // New Component
import HardwareSection from './components/HardwareSection';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <EnterLink />
      <div className="content">
        {/* Move ModelEmission inside EnterLink, do not put it here */}
        <div className="right-panel">
          <PipelineTimeline />
          <HardwareSection />
        </div>
      </div>
    </div>
  );
}

export default App;

