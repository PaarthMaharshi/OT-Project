import type { RunParams, RunProgress, RunResult } from './types';
import { useAuth } from './auth';

const authHeader = (): Record<string, string> => {
  const t = useAuth.getState().token;
  return t ? { Authorization: `Bearer ${t}` } : {};
};

// Requests go to /api, which Vite proxies to the Python server on port 8000 (see vite.config.ts).
const BASE = '/api';

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function backendAvailable(): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1500);
    const res = await fetch(`${BASE}/health`, { signal: ctrl.signal });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

/** Starts a run and polls it until it finishes, reporting progress along the way. */
export async function optimise(params: RunParams, onProgress: (p: RunProgress) => void): Promise<RunResult> {
  const { runId } = await json<{ runId: string }>(
    await fetch(`${BASE}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeader() }, body: JSON.stringify(params) }),
  );
  for (;;) {
    await new Promise((r) => setTimeout(r, 500));
    const st = await json<{ status: string; progress?: RunProgress; result?: RunResult; error?: string }>(await fetch(`${BASE}/runs/${runId}`, { headers: authHeader() }));
    if (st.progress) onProgress(st.progress);
    if (st.status === 'done' && st.result) return st.result;
    if (st.status === 'error') throw new Error(st.error ?? 'Optimiser failed');
  }
}
