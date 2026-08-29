import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { CaseListResponse } from '../types/api';
import PolicyBadge from '../components/PolicyBadge';

export default function RecoveryCases() {
  const [data, setData] = useState<CaseListResponse | null>(null);
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    setLoading(true);
    setError(null);
    const params: Record<string, unknown> = { page, page_size: 15 };
    if (filter) params.is_stop = filter === 'stop';
    api.cases(params as Parameters<typeof api.cases>[0])
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [page, filter]);

  if (error) return <div className="text-red-400 py-8">Error: {error}</div>;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Recovery Cases</h1>
        <div className="flex gap-2">
          <select
            value={filter}
            onChange={(e) => { setFilter(e.target.value); setPage(1); }}
            className="bg-gray-800 border border-gray-700 text-sm rounded px-3 py-1.5 text-gray-200"
          >
            <option value="">All</option>
            <option value="stop">STOP only</option>
          </select>
        </div>
      </div>

      {data && (
        <p className="text-sm text-gray-500">{data.total} cases total</p>
      )}

      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-left">
              <th className="px-4 py-2">Case ID</th>
              <th className="px-4 py-2">Amount</th>
              <th className="px-4 py-2">Failure</th>
              <th className="px-4 py-2">AI Rec</th>
              <th className="px-4 py-2">Authorized</th>
              <th className="px-4 py-2">ERV</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
            ) : data?.cases.map((c) => (
              <tr
                key={c.attempt_id}
                className="border-b border-gray-800/50 hover:bg-gray-800/50 cursor-pointer transition"
                onClick={() => navigate(`/cases/${c.attempt_id}`)}
              >
                <td className="px-4 py-2 font-mono text-xs text-gray-300">{c.attempt_id}</td>
                <td className="px-4 py-2">₹{c.amount_inr.toLocaleString()}</td>
                <td className="px-4 py-2 text-gray-400">{c.failure_category}</td>
                <td className="px-4 py-2"><PolicyBadge action={c.scoring_recommendation} size="sm" /></td>
                <td className="px-4 py-2"><PolicyBadge action={c.authorized_action} isStop={c.is_stop} size="sm" /></td>
                <td className="px-4 py-2 text-gray-300">₹{c.expected_recovery_value_inr.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data && (
        <div className="flex items-center justify-between text-sm">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-1 rounded bg-gray-800 text-gray-300 disabled:opacity-40 hover:bg-gray-700"
          >
            Previous
          </button>
          <span className="text-gray-500">
            Page {data.page} of {Math.ceil(data.total / data.page_size)}
          </span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={page * data.page_size >= data.total}
            className="px-3 py-1 rounded bg-gray-800 text-gray-300 disabled:opacity-40 hover:bg-gray-700"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
