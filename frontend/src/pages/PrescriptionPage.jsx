import { useState } from 'react';
import Eyebrow from '../components/Eyebrow.jsx';
import Alert from '../components/Alert.jsx';
import Expander from '../components/Expander.jsx';
import { analyzePrescription } from '../api.js';

const MOCK_RESULT = {
  drugs_detected: ['aspirin', 'warfarin', 'paracetamol'],
  interactions: [
    {
      drug_a: 'aspirin',
      drug_b: 'warfarin',
      severity: 'HIGH',
      description: 'Increases bleeding risk significantly',
    },
  ],
  interaction_severity: 0.85,
  risk_level: 'HIGH',
  dose_alerts: [],
  unresolved_dose_drugs: [],
};

function riskAlertVariant(risk) {
  if (risk === 'HIGH') return 'error';
  if (risk === 'MEDIUM') return 'warning';
  return 'success';
}

export default function PrescriptionPage() {
  const [prescription, setPrescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [mockNotice, setMockNotice] = useState(false);
  const [result, setResult] = useState(null);

  async function check() {
    if (!prescription.trim()) return;
    setMockNotice(false);
    setLoading(true);
    try {
      const data = await analyzePrescription(prescription);
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
      <Eyebrow>MODULE 02 - PRESCRIPTION NLP</Eyebrow>
      <h1>Prescription Safety Checker</h1>
      <p>Enter a prescription to check for dangerous drug interactions.</p>

      <textarea
        className="oncora-textarea"
        rows={6}
        placeholder={'e.g.\nParacetamol 500mg twice daily\nWarfarin 5mg\nAspirin 100mg'}
        value={prescription}
        onChange={(e) => setPrescription(e.target.value)}
      />

      <div style={{ marginTop: 14 }}>
        <button
          type="button"
          className="oncora-button"
          onClick={check}
          disabled={loading || !prescription.trim()}
        >
          {loading ? 'Analyzing...' : 'Check Prescription'}
        </button>
      </div>

      {mockNotice && <Alert variant="warning">Backend not connected - showing mock data</Alert>}

      {result && (
        <>
          <h3>Drugs Detected</h3>
          <p>{result.drugs_detected.length ? result.drugs_detected.join(', ') : 'None detected'}</p>

          <Alert variant={riskAlertVariant(result.risk_level)}>
            Risk Level: {result.risk_level}
          </Alert>

          {result.dose_alerts?.length > 0 && (
            <>
              <h3>Dose Safety Alerts</h3>
              {result.dose_alerts.map((alert, i) => {
                const label = `${titleCase(alert.drug)} - ${alert.limit_type} dose - ${alert.severity}`;
                return (
                  <div key={i}>
                    <Alert variant={alert.severity === 'HIGH' ? 'error' : 'warning'}>{label}</Alert>
                    <p>{alert.message}</p>
                  </div>
                );
              })}
            </>
          )}

          {result.unresolved_dose_drugs?.length > 0 && (
            <p className="oncora-caption">
              No dose reference data available for:{' '}
              {result.unresolved_dose_drugs.map(titleCase).join(', ')} - this does not mean the
              dose is safe, only that it wasn't checked.
            </p>
          )}

          {result.interactions.length > 0 ? (
            <>
              <h3>Interaction Alerts</h3>
              {result.interactions.map((interaction, i) => (
                <Expander
                  key={i}
                  title={`${titleCase(interaction.drug_a)} + ${titleCase(interaction.drug_b)} - ${interaction.severity}`}
                >
                  <p>{interaction.description}</p>
                </Expander>
              ))}
            </>
          ) : result.dose_alerts?.length > 0 ? (
            <Alert variant="info">No drug-drug interactions detected - see dose safety alerts above.</Alert>
          ) : (
            <Alert variant="success">No dangerous interactions or dose issues detected.</Alert>
          )}
        </>
      )}
    </div>
  );
}

function titleCase(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
