import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import RecoveryCases from './pages/RecoveryCases';
import CaseDetail from './pages/CaseDetail';
import PortfolioOptimizer from './pages/PortfolioOptimizer';
import RecoveryAgent from './pages/RecoveryAgent';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/cases" element={<RecoveryCases />} />
          <Route path="/cases/:caseId" element={<CaseDetail />} />
          <Route path="/portfolio" element={<PortfolioOptimizer />} />
          <Route path="/recovery-agent" element={<RecoveryAgent />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
