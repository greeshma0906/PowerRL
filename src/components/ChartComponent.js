import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from "recharts";

const data = [
  { epoch: 0, loss: 5.5, accuracy: 0.5 },
  { epoch: 1, loss: 4.8, accuracy: 0.6 },
  { epoch: 2, loss: 3.9, accuracy: 0.65 },
  { epoch: 3, loss: 3.2, accuracy: 0.7 },
  { epoch: 4, loss: 2.8, accuracy: 0.75 },
];

function ChartComponent() {
  return (
    <LineChart width={400} height={300} data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="epoch" label={{ value: "Epochs", position: "insideBottom", offset: -5 }} />
      <YAxis label={{ value: "Metrics", angle: -90, position: "insideLeft" }} />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey="loss" stroke="orange" dot={{ fill: "orange" }} />
      <Line type="monotone" dataKey="accuracy" stroke="green" dot={{ fill: "green" }} />
    </LineChart>
  );
}

export default ChartComponent;
