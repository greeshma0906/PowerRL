// import React from "react";
// import Header from "./components/Header";
// import EnterLink from "./components/EnterLink";
// //import PipelineTimeline from "./components/PipelineTimeline"; // New Component
// import HardwareSection from "./components/HardwareSection";
// import "./App.css";

// function App() {
//   return (
//     <div className="App">
//       <Header />
//       <EnterLink />
//       <div className="content">
//         {/* Move ModelEmission inside EnterLink, do not put it here */}
//         <div className="right-panel">

//         </div>
//       </div>
//     </div>
//   );
// }

// export default App;

import React from "react";
import Header from "./components/Header";
import EnterLink from "./components/EnterLink";
import "./App.css";

function App() {
  return (
    <div className="App">
      <Header />
      <EnterLink />
    </div>
  );
}

export default App;
