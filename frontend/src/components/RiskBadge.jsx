const COLORS = {
  LOW: '#1e5631',
  MODERATE: '#8a6d1f',
  HIGH: '#b31b1b',
};

export default function RiskBadge({ level }) {
  const upper = (level || '').toUpperCase();

  if (upper === 'CRITICAL') {
    return (
      <span
        className="oncora-risk-badge"
        style={{ color: '#fdfdfb', background: '#111111', borderColor: '#111111' }}
      >
        {upper}
      </span>
    );
  }

  const color = COLORS[upper] || '#111111';
  return (
    <span className="oncora-risk-badge" style={{ color }}>
      {upper}
    </span>
  );
}
