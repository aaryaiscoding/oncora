import { useState } from 'react';
import Eyebrow from '../components/Eyebrow.jsx';
import Alert from '../components/Alert.jsx';
import Metric from '../components/Metric.jsx';
import MriUploader from '../components/MriUploader.jsx';
import ProbabilityBarChart from '../components/ProbabilityBarChart.jsx';
import { predictMri } from '../api.js';

const TUMOR_INFO = [
  {
    title: 'Glioma',
    color: '#b31b1b',
    desc: "Originates in glial cells - the most common malignant brain tumor, accounting for ~80% of cases. Aggressive and fast-growing.",
  },
  {
    title: 'Meningioma',
    color: '#111111',
    desc: 'Arises from the meninges surrounding the brain and spinal cord. Usually benign and slow-growing, but can compress brain tissue.',
  },
  {
    title: 'Pituitary',
    color: '#8a6d1f',
    desc: "Develops in the pituitary gland at the brain's base. Typically non-cancerous, but disrupts hormone regulation.",
  },
  {
    title: 'No Tumor',
    color: '#1e5631',
    desc: 'No detectable malignancy found in the scan. All regions appear within normal parameters.',
  },
];

const MOCK_RESULT = {
  prediction: 'glioma',
  confidence: 0.91,
  all_probabilities: { glioma: 0.91, meningioma: 0.05, notumor: 0.03, pituitary: 0.01 },
};

export default function MriPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mockNotice, setMockNotice] = useState(false);
  const [result, setResult] = useState(null);

  async function analyze() {
    if (!file) {
      setError('Upload an MRI image first.');
      return;
    }
    setError(null);
    setMockNotice(false);
    setLoading(true);
    try {
      const data = await predictMri(file);
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
      <Eyebrow>MODULE 01 - DIAGNOSTIC IMAGING</Eyebrow>
      <h1>Brain MRI Classification</h1>
      <p>Upload an MRI scan to detect tumor type.</p>

      <div className="oncora-columns cols-2-3-2">
        <div>
          <MriUploader file={file} onChange={setFile} />

          <div style={{ marginTop: 14 }}>
            <button
              type="button"
              className="oncora-button"
              onClick={analyze}
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Scan'}
            </button>
          </div>

          {error && <Alert variant="error">{error}</Alert>}
          {mockNotice && <Alert variant="warning">Backend not connected - showing mock data</Alert>}

          {result && (
            <>
              <Alert variant="success">
                Prediction: <strong>{result.prediction.toUpperCase()}</strong>
              </Alert>

              <div className="oncora-metrics-row">
                <Metric label="Confidence" value={`${(result.confidence * 100).toFixed(1)}%`} />
                <Metric label="Status" value="Analysis complete" />
              </div>

              <h3>Class Probabilities</h3>
              <ProbabilityBarChart probabilities={result.all_probabilities} />
            </>
          )}
        </div>

        <div>
          <h3>Detectable Conditions</h3>
          {TUMOR_INFO.map(({ title, color, desc }) => (
            <div
              key={title}
              className="oncora-info-card"
              style={{ border: `1px solid ${color}28`, borderLeft: `3px solid ${color}`, background: `${color}0a` }}
            >
              <div className="oncora-info-card-title" style={{ color }}>{title}</div>
              <div className="oncora-info-card-desc">{desc}</div>
            </div>
          ))}

          <h3>Model Info</h3>
          <div className="oncora-metrics-row">
            <Metric label="Architecture" value="CNN" />
            <Metric label="Val Accuracy" value="91.0%" />
          </div>
        </div>
      </div>
    </div>
  );
}
