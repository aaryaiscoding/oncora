import { useEffect, useState } from 'react';
import Alert from '../components/Alert.jsx';
import Metric from '../components/Metric.jsx';
import Expander from '../components/Expander.jsx';
import AccuracyLineChart from '../components/AccuracyLineChart.jsx';
import { getNlpSanityHistory, runNlpSanityRound } from '../api.js';

export default function NlpSanityPage() {
  const [sampleSize, setSampleSize] = useState(10);
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [running, setRunning] = useState(false);
  const [runMessage, setRunMessage] = useState(null);

  async function loadHistory() {
    setLoadingHistory(true);
    try {
      const data = await getNlpSanityHistory();
      setHistory(data);
    } catch {
      setHistory([]);
    } finally {
      setLoadingHistory(false);
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function runRound() {
    setRunning(true);
    setRunMessage(null);
    try {
      const result = await runNlpSanityRound(sampleSize);
      setRunMessage({
        variant: 'success',
        text: `Round complete - ${(result.accuracy * 100).toFixed(1)}% overall accuracy (${result.n_samples} samples)`,
      });
      await loadHistory();
    } catch (e) {
      setRunMessage({ variant: 'error', text: `Sanity check failed: ${e.message}` });
    } finally {
      setRunning(false);
    }
  }

  const data = history.map((h, i) => ({
    round: i + 1,
    accuracy: h.accuracy,
    extraction: h.drug_extraction_accuracy,
    severity: h.severity_accuracy,
  }));

  const latest = history.length ? history[history.length - 1] : null;

  return (
    <div>
      <h1>Prescription NLP Sanity Check</h1>
      <p>
        Randomized, repeatable check of the drug-extraction and interaction pipeline against known
        ground-truth drug pairs — same round-based accuracy tracking as the Federated Monitor tab,
        applied here to the prescription NLP pipeline instead of the MRI model.
      </p>

      <div className="oncora-columns cols-2-1">
        <div>
          <label htmlFor="sample-size" style={{ fontSize: 13, color: '#333333' }}>
            Random drug pairs to sample this round: <strong>{sampleSize}</strong>
          </label>
          <br />
          <input
            id="sample-size"
            type="range"
            min={5}
            max={50}
            value={sampleSize}
            onChange={(e) => setSampleSize(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button type="button" className="oncora-button block" onClick={runRound} disabled={running}>
            {running ? 'Running...' : 'Run New Round'}
          </button>
        </div>
      </div>

      {runMessage && <Alert variant={runMessage.variant}>{runMessage.text}</Alert>}

      {!loadingHistory && history.length === 0 && (
        <Alert variant="warning">No sanity check history yet — run a round above to get started.</Alert>
      )}

      {history.length > 0 && (
        <>
          <h3>Accuracy Over Rounds</h3>
          <AccuracyLineChart
            data={data}
            showLegend
            series={[
              { key: 'accuracy', name: 'Overall Accuracy', color: '#4169e1' },
              { key: 'extraction', name: 'Drug Extraction Accuracy', color: '#2e8b57', dash: true },
              { key: 'severity', name: 'Severity Accuracy', color: '#ff8c00', dash: true },
            ]}
          />

          <div className="oncora-metrics-row">
            <Metric label="Rounds Run" value={String(history.length)} />
            <Metric label="Latest Overall Accuracy" value={`${(latest.accuracy * 100).toFixed(1)}%`} />
            <Metric label="Latest Extraction Accuracy" value={`${(latest.drug_extraction_accuracy * 100).toFixed(1)}%`} />
            <Metric
              label="Latest Severity Accuracy"
              value={latest.severity_accuracy != null ? `${(latest.severity_accuracy * 100).toFixed(1)}%` : 'N/A'}
            />
          </div>

          <h3>Most Recent Round — Failures</h3>
          {latest.failures?.length > 0 ? (
            <>
              <Alert variant="error">{latest.failures.length} failure(s) in the most recent round:</Alert>
              {latest.failures.map((f, i) => (
                <Expander key={i} title={`${titleCase(f.drug_a)} + ${titleCase(f.drug_b)} — ${f.issue}`}>
                  <pre style={{ fontSize: 12, whiteSpace: 'pre-wrap', margin: 0 }}>{JSON.stringify(f, null, 2)}</pre>
                </Expander>
              ))}
            </>
          ) : (
            <Alert variant="success">No failures in the most recent round.</Alert>
          )}
        </>
      )}
    </div>
  );
}

function titleCase(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
