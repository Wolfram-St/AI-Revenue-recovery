import type {
  HealthResponse,
  DashboardResponse,
  CaseListResponse,
  CaseDetailResponse,
  AnalysisResponse,
  PortfolioRequest,
  PortfolioResponse,
  AuditResponse,
} from '../types/api';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error?.message || `HTTP ${response.status}`);
  }
  return response.json();
}

export const api = {
  health: () => fetchJson<HealthResponse>('/health'),
  dashboard: () => fetchJson<DashboardResponse>('/api/dashboard'),
  cases: (params?: {
    page?: number;
    page_size?: number;
    failure_category?: string;
    recommendation?: string;
    is_stop?: boolean;
  }) => {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.set('page', String(params.page));
    if (params?.page_size) searchParams.set('page_size', String(params.page_size));
    if (params?.failure_category) searchParams.set('failure_category', params.failure_category);
    if (params?.recommendation) searchParams.set('recommendation', params.recommendation);
    if (params?.is_stop !== undefined) searchParams.set('is_stop', String(params.is_stop));
    const qs = searchParams.toString();
    return fetchJson<CaseListResponse>(`/api/cases${qs ? `?${qs}` : ''}`);
  },
  caseDetail: (caseId: string) =>
    fetchJson<CaseDetailResponse>(`/api/cases/${encodeURIComponent(caseId)}`),
  analyze: (caseId: string) =>
    fetchJson<AnalysisResponse>(`/api/cases/${encodeURIComponent(caseId)}/analyze`, {
      method: 'POST',
    }),
  optimize: (request: PortfolioRequest) =>
    fetchJson<PortfolioResponse>('/api/portfolio/optimize', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
  audit: (params?: { case_id?: string; page?: number; page_size?: number }) => {
    const searchParams = new URLSearchParams();
    if (params?.case_id) searchParams.set('case_id', params.case_id);
    if (params?.page) searchParams.set('page', String(params.page));
    if (params?.page_size) searchParams.set('page_size', String(params.page_size));
    const qs = searchParams.toString();
    return fetchJson<AuditResponse>(`/api/audit${qs ? `?${qs}` : ''}`);
  },
};
