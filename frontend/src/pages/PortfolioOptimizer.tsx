import { useState } from 'react';
import { api } from '../api/client';
import type { PortfolioResponse } from '../types/api';
import PolicyBadge from '../components/PolicyBadge';

export default function PortfolioOptimizer() {
  const [budget, setBudget] = useState(2000);
  const [hrCapacity, setHrCapacity] = useState(10);
  const [result, setResult] = useState<PortfolioResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.optimize({ budget_inr: budget, human_review_capacity: hrCapacity });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Optimization failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Portfolio Optimizer</h1>

      {/* Controls */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-xs uppercase text-gray-500 mb-3 tracking-wide">Optimization Constraints</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Budget (INR)</label>
            <input
              type="number"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-gray-200 text-sm"
              min={0}
              step={100}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Human Review Capacity</label>
            <input
              type="number"
              value={hrCapacity}
              onChange={(e) => setHrCapacity(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-gray-200 text-sm"
              min={0}
            />
          </div>
          <button
            onClick={runOptimize}
            disabled={loading}
            className="px-4 py-2 rounded bg-emerald-600 text-white font-medium hover:bg-emerald-500 disabled:opacity-50 transition"
          >
            {loading ? 'Optimizing...' : 'Optimize Portfolio'}
          </button>
        </div>
      </div>

      {error && <div className="text-red-400 text-sm bg-red-950/30 rounded p-3">{error}</div>}

      {result && (
        <>
          {/* Summary */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h2 className="text-xs uppercase text-gray-500 mb-3 tracking-wide">Solver Result</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-gray-800 rounded p-3 text-center">
                <p className="text-xs text-gray-500">Solver</p>
                <p className="font-mono text-gray-200">{result.solver}</p>
              </div>
              <div className="bg-gray-800 rounded p-3 text-center">
                <p className="text-xs text-gray-500">Expected Value</p>
                <p className="text-lg font-bold text-emerald-400">₹{result.summary.optimizer_objective_value_inr.toFixed(2)}</p>
              </div>
              <div className="bg-gray-800 rounded p-3 text-center">
                <p className="text-xs text-gray-500">Selected</p>
                <p className="text-lg font-bold text-blue-400">{result.summary.optimizer_allocated_count}</p>
              </div>
              <div className="bg-gray-800 rounded p-3 text-center">
                <p className="text-xs text-gray-500">Policy Overrides</p>
                <p className="text-lg font-bold text-amber-400">{result.summary.total_policy_overrides}</p>
              </div>
            </div>
          </div>

          {/* Budget & HR */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h2 className="text-xs uppercase text-gray-500 mb-3 tracking-wide">Allocation Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Budget Used:</span>{' '}
                <span className="text-gray-200">₹{result.summary.budget_allocated_inr.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-gray-500">Budget Remaining:</span>{' '}
                <span className="text-gray-200">{result.summary.budget_remaining_inr !== null ? `₹${result.summary.budget_remaining_inr.toFixed(2)}` : 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-500">HR Used:</span>{' '}
                <span className="text-gray-200">{result.summary.human_review_allocated_count}</span>
              </div>
              <div>
                <span className="text-gray-500">HR Remaining:</span>{' '}
                <span className="text-gray-200">{result.summary.human_review_capacity_limit !== null ? result.summary.human_review_capacity_limit - result.summary.human_review_allocated_count : 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* Action Distribution */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h2 className="text-xs uppercase text-gray-500 mb-3 tracking-wide">Action Distribution (Authorized)</h2>
            <div className="flex flex-wrap gap-3">
              {Object.entries(result.summary.action_authorized_counts).map(([action, count]) => (
                <div key={action} className="bg-gray-800 rounded px-3 py-2 text-center min-w-[80px]">
                  <p className="text-lg font-bold text-gray-100">{count}</p>
                  <p className="text-xs text-gray-500">{action}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Entries */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <div className="px-4 py-2 border-b border-gray-800">
              <h2 className="text-xs uppercase text-gray-500 tracking-wide">Selected Opportunities ({result.summary.optimizer_allocated_count})</h2>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-gray-500 text-left">
                  <th className="px-4 py-2">Case</th>
                  <th className="px-4 py-2">Optimizer Rec</th>
                  <th className="px-4 py-2">Authorized</th>
                  <th className="px-4 py-2">Net Value</th>
                  <th className="px-4 py-2">Override</th>
                </tr>
              </thead>
              <tbody>
                {result.entries
                  .filter((e) => e.optimizer_recommendation !== 'NO_INTERVENTION')
                  .slice(0, 50)
                  .map((e) => (
                    <tr key={e.attempt_id} className="border-b border-gray-800/50">
                      <td className="px-4 py-2 font-mono text-xs text-gray-300">{e.attempt_id}</td>
                      <td className="px-4 py-2"><PolicyBadge action={e.optimizer_recommendation} size="sm" /></td>
                      <td className="px-4 py-2"><PolicyBadge action={e.authorized_action} isStop={e.authorized_action === 'STOP'} size="sm" /></td>
                      <td className="px-4 py-2 text-gray-300">
                        {e.selected_net_incremental_value_inr !== null ? `₹${e.selected_net_incremental_value_inr.toFixed(2)}` : '-'}
                      </td>
                      <td className="px-4 py-2">
                        {e.policy_overrode_recommendation && (
                          <span className="text-amber-400 text-xs font-medium">OVERRIDE</span>
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!result && !loading && !error && (
        <div className="text-center py-12 text-gray-500">
          Configure constraints and click "Optimize Portfolio" to run the Day 7 optimizer.
        </div>
      )}
    </div>
  );
}
