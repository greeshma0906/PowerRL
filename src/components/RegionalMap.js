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
