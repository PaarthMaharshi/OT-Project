import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSelectedSolution, useStore } from '../store';
import { pickLabels, representativePicks, type PickKind } from '../mock/solutions';
import type { Objectives, Solution } from '../types';
import { ObjectiveBars } from './Dashboard';

const AX: Record<keyof Objectives, { label: string; fmt: (v: number) => string }> = {
  f1: { label: 'f1 · idle gaps', fmt: (v) => v.toFixed(1) },
  f2: { label: 'f2 · room wastage', fmt: (v) => `${(v * 100).toFixed(0)}%` },
  f3: { label: 'f3 · load imbalance', fmt: (v) => v.toFixed(2) },
};
const pickColor: Record<PickKind, string> = { gaps: '#c8553d', rooms: '#d9a441', balance: '#7a8b6f', knee: '#5b6fb5' };

function Scatter({
  xk,
  yk,
  all,
  picks,
  selectedId,
  showAll,
  onPick,
  baseline,
}: {
  xk: keyof Objectives;
  yk: keyof Objectives;
  all: Solution[];
  picks: Record<PickKind, Solution>;
  selectedId: string;
  showAll: boolean;
  onPick: (id: string) => void;
  baseline: Solution | null;
}) {
  const W = 340;
  const H = 280;
  const m = { l: 46, r: 14, t: 14, b: 40 };
  const pts = showAll ? all : all.filter((s) => s.onFront);
  const withBase = baseline ? [...pts, baseline] : pts;
  const xs = withBase.map((s) => s.objectives[xk]);
  const ys = withBase.map((s) => s.objectives[yk]);
  const pad = (lo: number, hi: number) => {
    const d = hi - lo || Math.abs(hi) || 1;
    return [lo - d * 0.08, hi + d * 0.08];
  };
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs));
  const [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const sx = (v: number) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (v: number) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const pickOf = (s: Solution) => (Object.keys(picks) as PickKind[]).find((k) => picks[k].id === s.id);
  const ordered = [...pts].sort((a, b) => Number(a.onFront) - Number(b.onFront));

  return (
    <svg className="scatter" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${AX[xk].label} vs ${AX[yk].label}`}>
      <rect x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} fill="#fbf7ef" rx="8" />
      {[0.25, 0.5, 0.75].map((t) => (
        <g key={t} stroke="#e8dfcf" strokeDasharray="3 4">
          <line x1={m.l} x2={W - m.r} y1={m.t + t * (H - m.t - m.b)} y2={m.t + t * (H - m.t - m.b)} />
          <line y1={m.t} y2={H - m.b} x1={m.l + t * (W - m.l - m.r)} x2={m.l + t * (W - m.l - m.r)} />
        </g>
      ))}
      <text x={m.l} y={H - m.b + 16} fontSize="10" fill="#a39b8e">{AX[xk].fmt(x0)}</text>
      <text x={W - m.r} y={H - m.b + 16} fontSize="10" fill="#a39b8e" textAnchor="end">{AX[xk].fmt(x1)}</text>
      <text x={(m.l + W - m.r) / 2} y={H - 6} fontSize="12" fill="#6b655c" textAnchor="middle" fontWeight="600">{AX[xk].label}</text>
      <text x={m.l - 6} y={H - m.b} fontSize="10" fill="#a39b8e" textAnchor="end">{AX[yk].fmt(y0)}</text>
      <text x={m.l - 6} y={m.t + 8} fontSize="10" fill="#a39b8e" textAnchor="end">{AX[yk].fmt(y1)}</text>
      <text transform={`translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)`} fontSize="12" fill="#6b655c" textAnchor="middle" fontWeight="600">
        {AX[yk].label}
      </text>
      {ordered.map((s) => {
        const pk = pickOf(s);
        const sel = s.id === selectedId;
        const fill = pk ? pickColor[pk] : s.onFront ? '#2b2a28' : '#cfc6b6';
        return (
          <g key={s.id} onClick={() => onPick(s.id)} style={{ cursor: 'pointer' }}>
            {sel && <circle cx={sx(s.objectives[xk])} cy={sy(s.objectives[yk])} r="10" fill="none" stroke="#2b2a28" strokeWidth="2" />}
            <circle className="pt" cx={sx(s.objectives[xk])} cy={sy(s.objectives[yk])} r={pk ? 6.5 : s.onFront ? 4.5 : 3} fill={fill} opacity={s.onFront ? 1 : 0.7}>
              <title>
                {s.id} — gaps {s.objectives.f1.toFixed(1)}, waste {(s.objectives.f2 * 100).toFixed(0)}%, imbalance {s.objectives.f3.toFixed(3)}
              </title>
            </circle>
          </g>
        );
      })}
      {baseline && (
        <g>
          <rect
            x={sx(baseline.objectives[xk]) - 6}
            y={sy(baseline.objectives[yk]) - 6}
            width="12"
            height="12"
            fill="#fff"
            stroke="#2b2a28"
            strokeWidth="2"
            transform={`rotate(45 ${sx(baseline.objectives[xk])} ${sy(baseline.objectives[yk])})`}
          >
            <title>
              Greedy baseline — gaps {baseline.objectives.f1.toFixed(1)}, waste {(baseline.objectives.f2 * 100).toFixed(0)}%, imbalance{' '}
              {baseline.objectives.f3.toFixed(3)}
            </title>
          </rect>
        </g>
      )}
    </svg>
  );
}

export default function ParetoExplorer() {
  const { solutions, selectedId, select, baseline } = useStore();
  const selected = useSelectedSolution();
  const nav = useNavigate();
  const [showAll, setShowAll] = useState(true);
  const front = useMemo(() => solutions.filter((s) => s.onFront).sort((a, b) => a.objectives.f1 - b.objectives.f1), [solutions]);
  const picks = useMemo(() => representativePicks(front), [front]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Pareto front</h1>
          <p>
            Every dot is a clash-free timetable. Dark dots are non-dominated: no other timetable beats them on all three
            objectives at once. Click any dot to select it.
          </p>
        </div>
        <div className="spacer" />
        <label className="check">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> Show dominated solutions
        </label>
      </div>

      <div className="row" style={{ marginBottom: 14, gap: 14 }}>
        {(Object.keys(pickLabels) as PickKind[]).map((k) => (
          <span key={k} className="row tiny" style={{ gap: 6, color: 'var(--ink-2)' }}>
            <i style={{ width: 10, height: 10, borderRadius: 99, background: pickColor[k], display: 'inline-block' }} /> {pickLabels[k].title}
          </span>
        ))}
        {baseline && (
          <span className="row tiny" style={{ gap: 6, color: 'var(--ink-2)' }}>
            <i style={{ width: 9, height: 9, border: '2px solid #2b2a28', background: '#fff', transform: 'rotate(45deg)', display: 'inline-block' }} /> Greedy
            baseline
          </span>
        )}
      </div>

      <div className="grid grid-3" style={{ marginBottom: 22 }}>
        {([['f1', 'f2'], ['f1', 'f3'], ['f2', 'f3']] as [keyof Objectives, keyof Objectives][]).map(([x, y]) => (
          <div key={x + y} className="card" style={{ padding: 12 }}>
            <Scatter xk={x} yk={y} all={solutions} picks={picks} selectedId={selectedId} showAll={showAll} onPick={select} baseline={baseline} />
          </div>
        ))}
      </div>

      <div className="grid split">
        <div className="card table-wrap">
          <h2 style={{ marginBottom: 10 }}>Non-dominated timetables</h2>
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Idle gaps</th>
                <th>Room waste</th>
                <th>Imbalance</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {front.map((s) => {
                const pk = (Object.keys(picks) as PickKind[]).filter((k) => picks[k].id === s.id);
                return (
                  <tr key={s.id} className={s.id === selectedId ? 'sel' : ''} onClick={() => select(s.id)}>
                    <td><b>{s.id}</b></td>
                    <td>{s.objectives.f1.toFixed(1)}</td>
                    <td>{(s.objectives.f2 * 100).toFixed(1)}%</td>
                    <td>{s.objectives.f3.toFixed(3)}</td>
                    <td>
                      {pk.map((k) => (
                        <span key={k} className="chip" style={{ background: pickColor[k], color: '#fff', marginRight: 4 }}>
                          {pickLabels[k].title}
                        </span>
                      ))}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="card tape pick">
          <h2>{selected.id}</h2>
          <span className="tiny">{selected.onFront ? 'On the Pareto front' : 'Dominated — another timetable is better on every objective'}</span>
          <ObjectiveBars s={selected} front={front} />
          <button className="btn accent" onClick={() => nav('/designer')}>
            Design this timetable →
          </button>
        </div>
      </div>
    </>
  );
}
