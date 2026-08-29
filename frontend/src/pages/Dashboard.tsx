import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { DashboardResponse } from '../types/api';
import MetricCard from '../components/MetricCard';

export default function Dashboard() {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.dashboard()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500 py-8">Loading dashboard...</div>;
  if (error) return <div className="text-red-400 py-8">Error: {error}</div>;
  if (!data) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        {data.demo_mode && (
          <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-1 rounded">
            Synthetic Data
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Revenue at Risk" value={`₹${data.revenue_at_risk_inr.toLocaleString()}`} color="red" />
        <MetricCard label="Est. Recoverable" value={`₹${data.estimated_recoverable_value_inr.toLocaleString()}`} color="emerald" />
        <MetricCard label="Active Cases" value={data.total_cases} color="blue" />
        <MetricCard label="Policy STOP" value={data.stop_count} color="amber" />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Action Distribution</h2>
        <div className="flex flex-wrap gap-3">
          {data.action_distribution.map((d) => (
            <div key={d.action} className="bg-gray-800 rounded px-3 py-2 text-center min-w-[100px]">
              <p className="text-lg font-bold text-gray-100">{d.count}</p>
              <p className="text-xs text-gray-500">{d.action}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-2">System Summary</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Candidates for intervention:</span>{' '}
            <span className="text-gray-200">{data.candidate_count}</span>
          </div>
          <div>
            <span className="text-gray-500">No-op (ERV ≤ 0):</span>{' '}
            <span className="text-gray-200">{data.noop_count}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
