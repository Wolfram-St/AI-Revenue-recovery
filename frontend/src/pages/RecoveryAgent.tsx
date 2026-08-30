import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type {
  RecoveryStatusResponse,
  RecoveryAuditEntry,
} from '../types/api';
import MetricCard from '../components/MetricCard';

export default function RecoveryAgent() {
  const [status, setStatus] = useState<RecoveryStatusResponse | null>(null);
  const [audit, setAudit] = useState<RecoveryAuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [simCount, setSimCount] = useState(20);
  const [budget, setBudget] = useState(1000);

  const load = async () => {
    try {
      const [s, a] = await Promise.all([
        api.recoveryStatus(),
        api.recoveryAudit(),
      ]);
      setStatus(s);
      setAudit(a);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const runSimulation = async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await api.recoverySimulate({
        count: simCount,
        budget_limit_inr: budget,
      });
      setStatus(result);
      const a = await api.recoveryAudit();
      setAudit(a);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  if (loading) return <div className="text-gray-500 py-8">Loading recovery agent...</div>;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Recovery Agent</h1>
          <p className="text-sm text-gray-500 mt-1">
            Autonomous detect → diagnose → decide → execute pipeline
          </p>
        </div>
        <span className={`text-xs px-2 py-1 rounded font-medium ${
          status?.mode === 'simulation'
            ? 'bg-amber-500/20 text-amber-400'
            : status?.mode === 'live'
            ? 'bg-red-500/20 text-red-400'
            : 'bg-gray-500/20 text-gray-400'
        }`}>
          {status?.mode === 'simulation' ? 'SIMULATION' : status?.mode === 'live' ? 'LIVE' : 'IDLE'}
        </span>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Controls */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Run Simulation</h2>
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Failures to generate</label>
            <input
              type="number"
              value={simCount}
              onChange={(e) => setSimCount(Number(e.target.value))}
              min={1}
              max={500}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 w-24"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Budget (INR)</label>
            <input
              type="number"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              min={100}
              step={100}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 w-28"
            />
          </div>
          <button
            onClick={runSimulation}
            disabled={running}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-1.5 rounded text-sm font-medium transition"
          >
            {running ? 'Running...' : 'Run Agent'}
          </button>
        </div>
      </div>

      {/* Metrics */}
      {status && status.total_processed > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <MetricCard
            label="Processed"
            value={String(status.total_processed)}
            color="blue"
          />
          <MetricCard
            label="Revenue at Risk"
            value={`Rs.${status.total_amount_inr.toLocaleString()}`}
            color="red"
          />
          <MetricCard
            label="Recovered"
            value={`Rs.${status.recovered_inr.toLocaleString()}`}
            color="emerald"
          />
          <MetricCard
            label="Escalated"
            value={String(status.escalated_count)}
            color="amber"
          />
          <MetricCard
            label="Budget Left"
            value={`Rs.${status.budget_remaining_inr.toLocaleString()}`}
            color="blue"
          />
        </div>
      )}

      {/* Recovery Rate Bar */}
      {status && status.total_processed > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-medium text-gray-400">Recovery Rate</h2>
            <span className="text-emerald-400 font-bold">{status.recovery_rate_pct}%</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-3">
            <div
              className="bg-emerald-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(status.recovery_rate_pct, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Action & Category Distribution */}
      {status && status.audit_summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h2 className="text-sm font-medium text-gray-400 mb-3">Actions Taken</h2>
            <div className="space-y-2">
              {status.audit_summary.action_distribution?.map((d) => (
                <div key={d.action} className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">{d.action}</span>
                  <span className="text-sm font-mono text-gray-500">{d.count}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h2 className="text-sm font-medium text-gray-400 mb-3">Root Causes</h2>
            <div className="space-y-2">
              {status.audit_summary.category_distribution?.map((d) => (
                <div key={d.category} className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">{d.category}</span>
                  <span className="text-sm font-mono text-gray-500">{d.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Audit Trail */}
      {audit.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-3">
            Audit Trail ({audit.length} entries)
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b border-gray-800">
                  <th className="pb-2 font-medium">Payment</th>
                  <th className="pb-2 font-medium">Amount</th>
                  <th className="pb-2 font-medium">Root Cause</th>
                  <th className="pb-2 font-medium">Action</th>
                  <th className="pb-2 font-medium">Status</th>
                  <th className="pb-2 font-medium">Reason</th>
                </tr>
              </thead>
              <tbody>
                {audit.map((entry) => (
                  <tr key={entry.entry_id} className="border-b border-gray-800/50">
                    <td className="py-2 font-mono text-gray-400">
                      {entry.payment_id.slice(0, 16)}...
                    </td>
                    <td className="py-2 text-gray-300">
                      Rs.{(entry.amount_paise / 100).toFixed(2)}
                    </td>
                    <td className="py-2">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        entry.failure_category === 'fraud_suspected'
                          ? 'bg-red-500/20 text-red-400'
                          : entry.failure_category === 'network_error'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-blue-500/20 text-blue-400'
                      }`}>
                        {entry.failure_category}
                      </span>
                    </td>
                    <td className="py-2">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        entry.action_type === 'retry_now'
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : entry.action_type === 'escalate_to_human'
                          ? 'bg-amber-500/20 text-amber-400'
                          : entry.action_type === 'stop'
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-gray-500/20 text-gray-400'
                      }`}>
                        {entry.action_type}
                      </span>
                    </td>
                    <td className="py-2">
                      {entry.execution_success ? (
                        <span className="text-emerald-400 text-xs">SUCCESS</span>
                      ) : (
                        <span className="text-red-400 text-xs">FAILED</span>
                      )}
                    </td>
                    <td className="py-2 text-gray-500 text-xs max-w-[200px] truncate">
                      {entry.strategy_reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty state */}
      {status && status.total_processed === 0 && !running && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-center">
          <p className="text-gray-500 text-sm">
            No recovery runs yet. Click <strong className="text-emerald-400">Run Agent</strong> to start a simulation.
          </p>
        </div>
      )}
    </div>
  );
}
