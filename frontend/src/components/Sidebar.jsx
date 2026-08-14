const PAGES = [
  'Unified Risk Score',
  'MRI Classification',
  'Prescription Safety',
  'Federated Monitor',
  'NLP Sanity Check',
];

export default function Sidebar({ page, onNavigate }) {
  return (
    <aside className="oncora-sidebar">
      <h1>ONCORA</h1>
      <div className="oncora-byline">
        Federated Learning for Clinical Diagnosis and Prescription Safety
      </div>

      <div className="oncora-nav-label">Navigate</div>
      <nav className="oncora-nav">
        {PAGES.map((p) => (
          <button
            key={p}
            type="button"
            className={`oncora-nav-item${p === page ? ' active' : ''}`}
            onClick={() => onNavigate(p)}
          >
            {p}
          </button>
        ))}
      </nav>

      <div className="oncora-footer">
        Aarya Mishra &middot; Aayushi Gupta &middot; Disha Madhusudhana
        <br />
        Academic prototype - not for clinical use
      </div>
    </aside>
  );
}
