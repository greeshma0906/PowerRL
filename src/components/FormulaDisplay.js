// import React from 'react';

// // A simple component to display mathematical formulas
// // In a real app, you'd use MathJax or KaTeX for proper rendering
// function FormulaDisplay({ formula }) {
//   return (
//     <div className="formula-display">
//       {formula}
//     </div>
//   );
// }

// export default FormulaDisplay;
import React from 'react';

function FormulaDisplay({ formula }) {
  return (
    <div className="formula-display">
      {formula}
    </div>
  );
}

export default FormulaDisplay;
