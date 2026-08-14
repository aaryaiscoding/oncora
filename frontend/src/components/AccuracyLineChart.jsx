import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * series: [{ key, name, color, dash? }]
 * data: [{ round: 1, [key]: value, ... }]
 */
export default function AccuracyLineChart({ data, series, height = 300, showLegend = false }) {
  return (
    <div className="oncora-chart-box" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e0" />
          <XAxis
            dataKey="round"
            label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 11 }}
            tick={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, fill: '#333333' }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, fill: '#333333' }}
          />
          <Tooltip
            contentStyle={{ fontFamily: 'IBM Plex Sans, sans-serif', fontSize: 12, border: '1px solid #111111', borderRadius: 2 }}
          />
          {showLegend && <Legend wrapperStyle={{ fontSize: 12, fontFamily: 'IBM Plex Sans, sans-serif' }} />}
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.name}
              stroke={s.color}
              strokeDasharray={s.dash ? '4 3' : undefined}
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
