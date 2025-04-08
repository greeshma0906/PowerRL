
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend
);

function ModelEmissions({ chartData }) {
  const options = {
    responsive: true,
    plugins: {
      legend: { display: true },
      title: { display: true, text: 'Carbon Emissions Over Epochs' }
    },
    scales: {
      x: { type: 'category', title: { display: true, text: 'Epoch' } },
      y: { type: 'linear', title: { display: true, text: 'Emissions (CO2 lbs)' } }
    }
  };

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto' }}>
      <Line data={chartData} options={options} />
    </div>
  );
}

export default ModelEmissions;



