export default function Metric({ label, value, delta }) {
  return (
    <div className="oncora-metric">
      <div className="oncora-metric-label">{label}</div>
      <div className="oncora-metric-value">{value}</div>
      {delta ? <div className="oncora-metric-delta">{delta}</div> : null}
    </div>
  );
}
