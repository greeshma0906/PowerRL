// import React from 'react';
// import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
// import 'leaflet/dist/leaflet.css';
// import usStatesData from '../data/us-states.json'; // You'll need to create this JSON file

// function RegionalMap() {
//   // Style function for the GeoJSON
//   const getStyleForState = (feature) => {
//     // Get CO2 value for the state (example values)
//     const stateMap = {
//       'Georgia': 0.65,
//       'Wyoming': 1.52,
//       // Add other states here...
//     };
    
//     const co2Value = stateMap[feature.properties.name] || 0.5;
    
//     // Color scale from light green to dark green
//     const getColor = (value) => {
//       return value > 1.5 ? '#1a6c1a' :
//              value > 1.2 ? '#238823' :
//              value > 0.9 ? '#2fa62f' :
//              value > 0.6 ? '#4bc04b' :
//              value > 0.3 ? '#77d077' :
//                           '#a3e0a3';
//     };
    
//     return {
//       fillColor: getColor(co2Value),
//       weight: 1,
//       opacity: 1,
//       color: 'white',
//       fillOpacity: 0.7
//     };
//   };

//   return (
//     <div className="map-container">
//       <div className="section-header">
//         <h3>Your Region's Energy Intensity <span>(Lower is Better)</span></h3>
//       </div>
      
//       <div className="map-wrapper">
//         <MapContainer 
//           center={[39.5, -98.35]} 
//           zoom={4} 
//           style={{ height: '350px', width: '100%' }}
//           zoomControl={false}
//           attributionControl={false}
//         >
//           <TileLayer
//             url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
//             attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
//           />
//           <GeoJSON 
//             data={usStatesData}
//             style={getStyleForState}
//           />
//         </MapContainer>
//       </div>
      
//       <div className="co2-scale">
//         <div className="scale-label">CO<sub>2</sub> lb / kWh</div>
//         <div className="color-scale">
//           <div className="scale-stop" style={{ backgroundColor: '#1a6c1a' }}></div>
//           <div className="scale-stop" style={{ backgroundColor: '#238823' }}></div>
//           <div className="scale-stop" style={{ backgroundColor: '#2fa62f' }}></div>
//           <div className="scale-stop" style={{ backgroundColor: '#4bc04b' }}></div>
//           <div className="scale-stop" style={{ backgroundColor: '#77d077' }}></div>
//           <div className="scale-stop" style={{ backgroundColor: '#a3e0a3' }}></div>
//         </div>
//         <div className="scale-values">
//           <span>1.6</span>
//           <span>1.4</span>
//           <span>1.2</span>
//           <span>0.8</span>
//           <span>0.4</span>
//           <span>0.0</span>
//         </div>
//       </div>
      
//       <div className="region-examples">
//         <div className="region-item">
//           <span className="region-name">Georgia</span>
//           <span className="region-value">0.65 CO<sub>2</sub> lb / kWh</span>
//         </div>
//         <div className="region-item">
//           <span className="region-name">Wyoming</span>
//           <span className="region-value">1.52 CO<sub>2</sub> lb / kWh</span>
//         </div>
//       </div>
//     </div>
//   );
// }

// export default RegionalMap;

import React from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

function RegionalMap() {
  return (
    <div>
      <h3>Your Region's Energy Intensity</h3>
      <MapContainer center={[37.8, -96]} zoom={4} style={{ height: '300px', width: '100%' }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      </MapContainer>
    </div>
  );
}

export default RegionalMap;
