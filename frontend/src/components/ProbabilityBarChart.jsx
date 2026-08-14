import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function ProbabilityBarChart({ probabilities }) {
  const data = Object.entries(probabilities || {}).map(([name, value]) => ({ name, value }));

  return (
    <div className="oncora-chart-box" style={{ height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e0" />
          <XAxis
            dataKey="name"
            tick={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: 11, fill: '#333333' }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, fill: '#333333' }}
          />
          <Tooltip
            contentStyle={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: 12, border: '1px solid #111111', borderRadius: 2 }}
            formatter={(value) => value.toFixed(4)}
          />
          <Bar dataKey="value" fill="#b31b1b" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
