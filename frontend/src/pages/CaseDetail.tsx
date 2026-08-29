import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../api/client';
import type { AnalysisResponse, CaseDetailResponse } from '../types/api';
import PolicyBadge from '../components/PolicyBadge';

export default function CaseDetail() {
  const { caseId } = useParams<{ caseId: string }>();
  const [detail, setDetail] = useState<CaseDetailResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId) return;
    setLoading(true);
    api.caseDetail(caseId)
      .then(setDetail)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [caseId]);

  const runAnalysis = async () => {
    if (!caseId) return;
    setAnalyzing(true);
    try {
      const result = await api.analyze(caseId);
      setAnalysis(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  if (loading) return <div className="text-gray-500 py-8">Loading case...</div>;
  if (error && !detail) return <div className="text-red-400 py-8">Error: {error}</div>;
  if (!detail) return null;

  const c = detail.case;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link to="/cases" className="text-gray-500 hover:text-gray-300 text-sm">← Cases</Link>
        <h1 className="text-2xl font-bold">Decision Explorer</h1>
      </div>

      {/* Failed Payment Event */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-xs uppercase text-gray-500 mb-3 tracking-wide">Payment Failure Event</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div><span className="text-gray-500">Case:</span> <span className="font-mono text-gray-300">{c.attempt_id}</span></div>
          <div><span className="text-gray-500">Amount:</span> <span className="text-gray-200">₹{c.amount_inr.toLocaleString()}</span></div>
          <div><span className="text-gray-500">Category:</span> <span className="text-gray-200">{c.failure_category}</span></div>
          <div><span className="text-gray-500">P(recovered):</span> <span className="text-gray-200">{(c.recovery_probability * 100).toFixed(1)}%</span></div>
        </div>
      </div>

      {/* AI Analysis */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs uppercase text-gray-500 tracking-wide">AI Recovery Analysis</h2>
          <button
            onClick={runAnalysis}
            disabled={analyzing}
            className="px-3 py-1 rounded bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 disabled:opacity-50 transition"
          >
            {analyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <div><span className="text-gray-500">Scoring:</span> <PolicyBadge action={c.scoring_recommendation} size="sm" /></div>
          <div><span className="text-gray-500">ERV:</span> <span className="text-gray-200">₹{c.expected_recovery_value_inr.toFixed(2)}</span></div>
          <div><span className="text-gray-500">Worth Intervening:</span> <span className={c.scoring_recommendation === 'INTERVENE' ? 'text-emerald-400' : 'text-gray-500'}>{c.scoring_recommendation === 'INTERVENE' ? 'Yes' : 'No'}</span></div>
        </div>
      </div>

      {/* Analysis Results */}
      {analysis && (
        <>
          {/* Candidate Actions */}
          {analysis.candidate_actions.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h2 className="text-xs uppercase text-gray-500 mb-3 tracking-wide">Candidate Actions</h2>
              <div className="space-y-2">
                {analysis.candidate_actions.map((a) => (
                  <div key={a.arm} className="flex items-center justify-between bg-gray-800 rounded px-3 py-2 text-sm">
                    <span className="text-gray-300 font-medium">{a.arm}</span>
                    <span className="text-gray-400">{(a.probability * 100).toFixed(1)}% recovery</span>
                    <span className="text-emerald-400">₹{a.expected_recovery_value_inr.toFixed(2)} ERV</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Policy Gate */}
          <div className={`border-2 rounded-lg p-4 ${analysis.policy.is_stop ? 'bg-red-950/30 border-red-500/50' : 'bg-gray-900 border-gray-800'}`}>
            <h2 className="text-xs uppercase tracking-wide mb-3 text-center font-bold text-amber-400">
              ═══════════ POLICY GATE ═══════════
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <p className="text-gray-500 text-xs mb-1">AI Recommendation</p>
                <PolicyBadge action={analysis.scoring_recommendation} />
              </div>
              <div className="text-center">
                <p className="text-gray-500 text-xs mb-1">Policy Authorization</p>
                <PolicyBadge action={analysis.policy.authorized_action} isStop={analysis.policy.is_stop} />
              </div>
              <div className="text-center">
                <p className="text-gray-500 text-xs mb-1">Final Authorized Action</p>
                <PolicyBadge action={analysis.policy.authorized_action} isStop={analysis.policy.is_stop} />
              </div>
            </div>
            <div className="mt-3 text-center text-xs text-gray-400">
              <p>Reason: {analysis.policy.reason}</p>
              {analysis.policy.matched_rule_id && (
                <p className="mt-1">Rule: {analysis.policy.matched_rule_id} ({analysis.policy.matched_rule_name})</p>
              )}
            </div>
          </div>

          {/* Audit */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h2 className="text-xs uppercase text-gray-500 mb-3 tracking-wide">Evaluated Rules</h2>
            <div className="flex flex-wrap gap-2">
              {analysis.audit_context.evaluated_rules.map((r) => (
                <span
                  key={r.rule_id}
                  className={`text-xs px-2 py-0.5 rounded border ${
                    r.matched
                      ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                      : 'bg-gray-800 text-gray-500 border-gray-700'
                  }`}
                >
                  {r.rule_id} {r.matched ? '✓' : '✗'}
                </span>
              ))}
            </div>
          </div>
        </>
      )}

      {error && <div className="text-red-400 text-sm">{error}</div>}
    </div>
  );
}
