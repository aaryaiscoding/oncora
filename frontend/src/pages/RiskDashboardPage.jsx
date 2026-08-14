import { useState } from 'react';
import Eyebrow from '../components/Eyebrow.jsx';
import Alert from '../components/Alert.jsx';
import Metric from '../components/Metric.jsx';
import RiskBadge from '../components/RiskBadge.jsx';
import MriUploader from '../components/MriUploader.jsx';
import ProbabilityBarChart from '../components/ProbabilityBarChart.jsx';
import { assessRisk } from '../api.js';

const MOCK_RESULT = {
  final_risk_score: 0.62,
  risk_level: 'HIGH',
  data_completeness: 'incomplete',
  breakdown: { tumor_risk_contribution: 0.85, interaction_severity: 0.55, dose_anomaly_score: 0.0 },
  mri_prediction: 'glioma',
  mri_confidence: 0.85,
  mri_all_probabilities: { glioma: 0.85, meningioma: 0.09, notumor: 0.04, pituitary: 0.02 },
  drugs_detected: ['aspirin', 'warfarin'],
  drug_sources: [
    { drug: 'aspirin', source: 'vocabulary', confidence: 1.0 },
    { drug: 'warfarin', source: 'vocabulary', confidence: 1.0 },
  ],
  unresolved_interaction_pairs: [],
  unresolved_dose_drugs: [],
  has_low_confidence_interaction_data: false,
  biobert_available: true,
};

function BreakdownBar({ label, value, weightLabel, color }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="oncora-breakdown-row">
      <div className="oncora-breakdown-head">
        <span>
          {label} <span className="weight">({weightLabel})</span>
        </span>
        <span className="value">{value.toFixed(3)}</span>
      </div>
      <div className="oncora-breakdown-track">
        <div className="oncora-breakdown-fill" style={{ background: color, width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function RiskDashboardPage() {
  const [file, setFile] = useState(null);
  const [prescriptionText, setPrescriptionText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mockNotice, setMockNotice] = useState(false);
  const [result, setResult] = useState(null);

  async function run() {
    setError(null);
    if (!file) {
      setError('Upload an MRI image first.');
      return;
    }
    if (!prescriptionText.trim()) {
      setError('Enter prescription text first.');
      return;
    }
    setMockNotice(false);
    setLoading(true);
    try {
      const data = await assessRisk(file, prescriptionText);
      setResult(data);
    } catch {
      setResult(MOCK_RESULT);
      setMockNotice(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <Eyebrow>MODULE 04 - INTEGRATION LAYER</Eyebrow>
      <h1>Unified Risk Assessment</h1>
      <p>
        Fuses tumor classification risk with prescription interaction and dose-safety risk into a
        single score: <code>0.5 x tumor_risk + 0.3 x interaction_severity + 0.2 x dose_anomaly_score</code>
      </p>

      <div className="oncora-columns cols-2">
        <div>
          <h3>MRI Scan</h3>
          <MriUploader file={file} onChange={setFile} />
        </div>
        <div>
          <h3>Prescription Text</h3>
          <textarea
            className="oncora-textarea"
            rows={8}
            placeholder="e.g. Aspirin 100mg daily. Warfarin 5mg once a day."
            value={prescriptionText}
            onChange={(e) => setPrescriptionText(e.target.value)}
          />
        </div>
      </div>

      <div style={{ marginTop: 14 }}>
        <button type="button" className="oncora-button block" onClick={run} disabled={loading}>
          {loading ? 'Running assessment...' : 'Run Unified Assessment'}
        </button>
      </div>

      {error && <Alert variant="error">{error}</Alert>}
      {mockNotice && <Alert variant="warning">Backend not connected - showing mock data</Alert>}

      {result && (
        <>
          <hr />

          <div className="oncora-columns cols-2-1">
            <div>
              <div className="oncora-score-label">Final Risk Score</div>
              <div className="oncora-score-value">{result.final_risk_score.toFixed(3)}</div>
              <RiskBadge level={result.risk_level} />
            </div>
            <div>
              {result.data_completeness === 'incomplete' ? (
                <Alert variant="warning">
                  <strong>Incomplete data</strong> - some drug pairs or doses could not be checked.
                  This score reflects only what was resolvable, not a confirmed absence of risk.
                </Alert>
              ) : (
                <Alert variant="success">All interaction and dose data resolved for this assessment.</Alert>
              )}
            </div>
          </div>

          <h3>Score Breakdown</h3>
          <BreakdownBar label="Tumor Risk" value={result.breakdown.tumor_risk_contribution} weightLabel="weight 0.5" color="#b31b1b" />
          <BreakdownBar label="Interaction Severity" value={result.breakdown.interaction_severity} weightLabel="weight 0.3" color="#111111" />
          <BreakdownBar label="Dose Anomaly" value={result.breakdown.dose_anomaly_score} weightLabel="weight 0.2" color="#8a6d1f" />

          <hr />

          <div className="oncora-columns cols-2">
            <div>
              <h3>MRI Finding</h3>
              <div className="oncora-metrics-row">
                <Metric label="Prediction" value={result.mri_prediction.toUpperCase()} />
                <Metric label="Confidence" value={`${(result.mri_confidence * 100).toFixed(1)}%`} />
              </div>
              <ProbabilityBarChart probabilities={result.mri_all_probabilities} />
            </div>

            <div>
              <h3>Drugs Detected</h3>
              {(result.drug_sources || []).map((src, i) => (
                <div key={i} className="oncora-drug-row">
                  <span className="oncora-drug-name">{src.drug}</span>
                  <span
                    className="oncora-drug-source"
                    style={{ color: src.source === 'vocabulary' ? '#111111' : '#b31b1b' }}
                  >
                    {src.source} - {src.confidence.toFixed(2)}
                  </span>
                </div>
              ))}
              {!result.biobert_available && (
                <p className="oncora-caption">BioBERT fallback unavailable this run - vocabulary matching only.</p>
              )}
            </div>
          </div>

          {(result.unresolved_interaction_pairs?.length > 0 || result.unresolved_dose_drugs?.length > 0) && (
            <>
              <hr />
              <h3>Unresolved - Not Checked, Not Confirmed Safe</h3>
              {result.unresolved_interaction_pairs?.length > 0 && (
                <p>
                  <strong>Interaction pairs:</strong>{' '}
                  {result.unresolved_interaction_pairs.map(([a, b]) => `${a} + ${b}`).join(', ')}
                </p>
              )}
              {result.unresolved_dose_drugs?.length > 0 && (
                <p>
                  <strong>Dose limits:</strong> {result.unresolved_dose_drugs.join(', ')}
                </p>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
