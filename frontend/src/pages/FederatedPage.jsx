import { useEffect, useState } from 'react';
import Eyebrow from '../components/Eyebrow.jsx';
import Alert from '../components/Alert.jsx';
import Metric from '../components/Metric.jsx';
import AccuracyLineChart from '../components/AccuracyLineChart.jsx';
import { getFlHistory } from '../api.js';

const MOCK_ACCURACY = [0.78, 0.86, 0.73, 0.9, 0.9];

export default function FederatedPage() {
  const [accuracy, setAccuracy] = useState(null);
  const [mockNotice, setMockNotice] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const history = await getFlHistory();
        if (cancelled) return;
        if (history.length > 0) {
          setAccuracy(history.map((entry) => entry.accuracy));
        } else {
          setAccuracy(MOCK_ACCURACY);
          setMockNotice(true);
        }
      } catch {
        if (cancelled) return;
        setAccuracy(MOCK_ACCURACY);
        setMockNotice(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const data = (accuracy || []).map((value, i) => ({ round: i + 1, accuracy: value }));

  return (
    <div>
      <Eyebrow>MODULE 03 - FEDERATED TRAINING</Eyebrow>
      <h1>Federated Learning Monitor</h1>
      <p>Live view of simulated hospital training nodes.</p>

      <h3>Hospital Nodes</h3>
      <div className="oncora-metrics-row">
        <Metric label="Hospital A" value="Active" delta="Local training complete" />
        <Metric label="Hospital B" value="Active" delta="Weights submitted" />
        <Metric label="Hospital C" value="Active" delta="Awaiting global model" />
      </div>

      <h3>Global Model Accuracy per Round</h3>
      {mockNotice && <Alert variant="warning">FL history file not found - showing mock data</Alert>}

      {accuracy && (
        <>
          <AccuracyLineChart data={data} series={[{ key: 'accuracy', name: 'Federated Accuracy', color: '#111111' }]} />

          <div className="oncora-metrics-row">
            <Metric label="Total Rounds" value={String(accuracy.length)} />
            <Metric label="Final Global Accuracy" value={`${(accuracy[accuracy.length - 1] * 100).toFixed(1)}%`} />
            <Metric label="Hospitals Participating" value="3" />
          </div>
        </>
      )}
    </div>
  );
}
