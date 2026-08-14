export const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export async function predictMri(file) {
  const form = new FormData();
  form.append('file', file, file.name || 'mri.jpg');
  const res = await fetch(`${API_URL}/predict-mri`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`predict-mri failed: ${res.status}`);
  return res.json();
}

export async function analyzePrescription(text) {
  const res = await fetch(`${API_URL}/analyze-prescription`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`analyze-prescription failed: ${res.status}`);
  return res.json();
}

export async function assessRisk(file, prescriptionText) {
  const form = new FormData();
  form.append('file', file, file.name || 'mri.jpg');
  form.append('prescription_text', prescriptionText);
  const res = await fetch(`${API_URL}/assess-risk`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`assess-risk failed: ${res.status}`);
  return res.json();
}

export async function getFlHistory() {
  const res = await fetch(`${API_URL}/fl-history`);
  if (!res.ok) throw new Error(`fl-history failed: ${res.status}`);
  return res.json();
}

export async function getNlpSanityHistory() {
  const res = await fetch(`${API_URL}/nlp-sanity-history`);
  if (!res.ok) throw new Error(`nlp-sanity-history failed: ${res.status}`);
  return res.json();
}

export async function runNlpSanityRound(sampleSize) {
  const res = await fetch(`${API_URL}/nlp-sanity-run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sample_size: sampleSize }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `nlp-sanity-run failed: ${res.status}`);
  }
  return res.json();
}
