import React from 'react';
import './EnterLink.css'; // Import CSS for styling

function EnterLink() {
  return (
    <div className="enter-link-container">
      <h3 className="enter-link-heading">Enter the link to your RL pipeline:</h3>
      <div className="enter-link-bar">
        <input
          type="text"
          placeholder="Paste your link here..."
          className="link-input"
        />
        <button className="measure-button">Measure</button>
      </div>
    </div>
  );
}

export default EnterLink;
