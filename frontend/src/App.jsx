import { useState } from 'react';
import Sidebar from './components/Sidebar.jsx';
import RiskDashboardPage from './pages/RiskDashboardPage.jsx';
import MriPage from './pages/MriPage.jsx';
import PrescriptionPage from './pages/PrescriptionPage.jsx';
import FederatedPage from './pages/FederatedPage.jsx';
import NlpSanityPage from './pages/NlpSanityPage.jsx';

const PAGE_COMPONENTS = {
  'Unified Risk Score': RiskDashboardPage,
  'MRI Classification': MriPage,
  'Prescription Safety': PrescriptionPage,
  'Federated Monitor': FederatedPage,
  'NLP Sanity Check': NlpSanityPage,
};

export default function App() {
  const [page, setPage] = useState('Unified Risk Score');
  const PageComponent = PAGE_COMPONENTS[page];

  return (
    <div className="oncora-app">
      <Sidebar page={page} onNavigate={setPage} />
      <main className="oncora-main">
        <PageComponent />
      </main>
    </div>
  );
}
