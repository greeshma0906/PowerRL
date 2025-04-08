// import React from 'react';

// function Header({ showAlternatives, setShowAlternatives }) {
//   return (
//     <div className="header-container">
//       <div className="header">
//         <h1>EnergyVis <span>Interactive Energy Tracking for RL Pipeline</span></h1>
//       </div>
      
//       {showAlternatives && (
//         <div className="notification-bar">
//           <i className="fa fa-info-circle"></i>
//           You're currently exploring alternatives! Click reset to reset alternatives with default values.
//           <button 
//             className="reset-btn"
//             onClick={() => setShowAlternatives(false)}
//           >
//             RESET
//           </button>
//         </div>
//       )}
//     </div>
//   );
// }

// export default Header;
import React from 'react';

function Header() {
  return (
    <header className="header">
      <h1>PowerRL</h1>c
      <p>Interactive Energy Tracking for RL Pipeline</p>
    </header>
  );

}

export default Header;
