import { useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useSelectedSolution, useStore } from '../store';
import { pickLabels, representativePicks, type PickKind } from '../mock/solutions';
import type { RunStats, Scenario, Solution } from '../types';
import TimetablePoster from '../components/TimetablePoster';
import PosterFrame from '../components/PosterFrame';
import { aspects } from '../themes/designs';

export function ObjectiveBars({ s, front }: { s: Solution; front: Solution[] }) {
  const rows = [
    { k: 'f1' as const, label: 'Idle gaps', color: 'var(--accent)', fmt: (v: number) => v.toFixed(1) },
    { k: 'f2' as const, label: 'Room waste', color: 'var(--mustard)', fmt: (v: number) => `${(v * 100).toFixed(0)}%` },
    { k: 'f3' as const, label: 'Imbalance', color: 'var(--sage)', fmt: (v: number) => v.toFixed(3) },
  ];
  return (
    <div className="bars">
      {rows.map((r) => {
        const vals = front.map((x) => x.objectives[r.k]);
        const lo = Math.min(...vals);
        const hi = Math.max(...vals);
        const t = hi === lo ? 0.5 : (s.objectives[r.k] - lo) / (hi - lo);
        return (
          <div key={r.k} className="bar">
            <span>{r.label}</span>
            <div className="track">
              <div className="fill" style={{ width: `${8 + Math.max(0, Math.min(1, t)) * 92}%`, background: r.color }} />
            </div>
            <b>{r.fmt(s.objectives[r.k])}</b>
          </div>
        );
      })}
    </div>
  );
}

const SCENARIOS: { id: Scenario; name: string; size: string; gens: number }[] = [
  { id: 'demo', name: 'Demo', size: '34 classes', gens: 40 },
  { id: 'small', name: 'Small', size: '~40 classes', gens: 40 },
  { id: 'medium', name: 'Medium', size: '~150 classes', gens: 30 },
  { id: 'large', name: 'Large', size: '~400 classes', gens: 15 },
];

function Convergence({ stats }: { stats: RunStats }) {
  const pts = stats.convergence;
  if (pts.length < 2) return null;
  const W = 260;
  const H = 70;
  const max = Math.max(...pts.map((p) => p.hypervolume), stats.baselineHypervolume, 1e-9);
  const x = (i: number) => (i / (pts.length - 1)) * W;
  const y = (v: number) => H - 4 - (v / max) * (H - 10);
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(p.hypervolume).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="Hypervolume per generation">
      <line x1="0" x2={W} y1={y(stats.baselineHypervolume)} y2={y(stats.baselineHypervolume)} stroke="#a39b8e" strokeDasharray="4 4" />
      <text x={W} y={y(stats.baselineHypervolume) - 4} fontSize="9" fill="#a39b8e" textAnchor="end">
        greedy
      </text>
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinejoin="round" />
    </svg>
  );
}

function RunPanel() {
  const { runParams, setRunParams, runOptimizer, running, progress, runError, source, stats } = useStore();
  const pct = progress && progress.total ? Math.round((progress.generation / progress.total) * 100) : 0;

  return (
    <section className="card run-panel" style={{ marginBottom: 22 }}>
      <div className="run-grid">
        <div className="run-controls">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <h2>Optimiser</h2>
            <span className={`chip ${source === 'backend' ? 'sage' : 'accent'}`}>{source === 'backend' ? '● Live results' : '○ Demo data'}</span>
          </div>
          <div>
            <div className="section-title">Problem size</div>
            <div className="scenario-row">
              {SCENARIOS.map((s) => (
                <button
                  key={s.id}
                  className={`scenario ${runParams.scenario === s.id ? 'on' : ''}`}
                  onClick={() => setRunParams({ scenario: s.id, generations: s.gens })}
                  disabled={running}
                >
                  <b>{s.name}</b>
                  <span>{s.size}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="row" style={{ gap: 18, alignItems: 'flex-end' }}>
            <label className="field" style={{ flex: 1, minWidth: 160 }}>
              Generations: {runParams.generations}
              <input type="range" min={5} max={150} step={5} value={runParams.generations} disabled={running} onChange={(e) => setRunParams({ generations: +e.target.value })} />
            </label>
            <label className="field" style={{ flex: 1, minWidth: 160 }}>
              Population: {runParams.popSize}
              <input type="range" min={20} max={150} step={10} value={runParams.popSize} disabled={running} onChange={(e) => setRunParams({ popSize: +e.target.value })} />
            </label>
          </div>
          <details className="advanced">
            <summary>Advanced: seeds and hybrid steps</summary>
            <div className="row" style={{ gap: 14, marginTop: 10 }}>
              <label className="field" style={{ width: 110 }}>
                Instance seed
                <input className="input" type="number" value={runParams.instanceSeed} disabled={running} onChange={(e) => setRunParams({ instanceSeed: +e.target.value || 0 })} />
              </label>
              <label className="field" style={{ width: 110 }}>
                Run seed
                <input className="input" type="number" value={runParams.seed} disabled={running} onChange={(e) => setRunParams({ seed: +e.target.value || 0 })} />
              </label>
            </div>
            <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
              <label className="check">
                <input type="checkbox" checked={runParams.useSeeding} disabled={running} onChange={(e) => setRunParams({ useSeeding: e.target.checked })} />
                Seeding: start from Greedy/DSatur timetables
              </label>
              <label className="check">
                <input type="checkbox" checked={runParams.useRepair} disabled={running} onChange={(e) => setRunParams({ useRepair: e.target.checked })} />
                Repair: fix clashes in every child
              </label>
              <label className="check">
                <input type="checkbox" checked={runParams.useLocalSearch} disabled={running} onChange={(e) => setRunParams({ useLocalSearch: e.target.checked })} />
                Local search: compact gaps, best-fit rooms
              </label>
            </div>
          </details>
          <div className="row">
            <button className="btn accent" onClick={runOptimizer} disabled={running}>
              {running ? <span className="spinner" /> : '▶'} {running ? 'Optimising…' : 'Run optimiser'}
            </button>
            {running && progress && (
              <span className="tiny">
                Generation {progress.generation} / {progress.total}
                {progress.archiveSize != null && ` · ${progress.archiveSize} on the front`}
                {progress.seconds != null && ` · ${progress.seconds.toFixed(1)}s`}
              </span>
            )}
          </div>
          {running && (
            <div className="progress">
              <div style={{ width: `${pct}%` }} />
            </div>
          )}
          {runError && <div className="error-box">{runError}</div>}
        </div>

        <div className="run-stats">
          {stats ? (
            <>
              <div className="section-title">Last run</div>
              <div className="mini-stats">
                <div>
                  <b>{stats.frontSize}</b>
                  <span>timetables on the front</span>
                </div>
                <div>
                  <b>{stats.runtime.toFixed(1)}s</b>
                  <span>{stats.evaluations.toLocaleString()} evaluations</span>
                </div>
                <div>
                  <b>{stats.hypervolume.toFixed(2)}</b>
                  <span>hypervolume (greedy {stats.baselineHypervolume.toFixed(2)})</span>
                </div>
                <div>
                  <b>{stats.generations}</b>
                  <span>generations</span>
                </div>
              </div>
              <div className="section-title" style={{ marginTop: 14 }}>
                Hypervolume per generation
              </div>
              <Convergence stats={stats} />
            </>
          ) : (
            <div className="empty-hint" style={{ height: '100%', display: 'grid', placeItems: 'center' }}>
              Run the optimiser to see the hypervolume, run time and convergence here.
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default function Dashboard() {
  const { solutions, select, selectedId, theme, view, decorations, blockNotes, instance: inst } = useStore();
  const selected = useSelectedSolution();
  const nav = useNavigate();
  const front = useMemo(() => solutions.filter((s) => s.onFront), [solutions]);
  const picks = useMemo(() => representativePicks(front), [front]);
  const size = aspects.find((a) => a.id === theme.aspect)!;

  const tiles = [
    { n: inst.groups.length, label: 'Student groups', bg: 'var(--accent-soft)' },
    { n: inst.events.length, label: 'Weekly classes', bg: '#f6e3b8' },
    { n: inst.rooms.length, label: 'Rooms & labs', bg: 'var(--sage-soft)' },
    { n: inst.faculty.length, label: 'Faculty', bg: '#dfe3f3' },
  ];

  return (
    <>
      <div className="page-head">
        <p style={{ marginTop: 0 }}>
          Clash-free timetables trading off idle gaps, room wastage and workload balance. Pick one of the recommended
          timetables, explore the full Pareto front, then style it in the Designer.
        </p>
      </div>

      <RunPanel />

      <div className="grid grid-4" style={{ marginBottom: 22 }}>
        {tiles.map((s) => (
          <div key={s.label} className="card stat">
            <div className="dot" style={{ background: s.bg }} />
            <b>{s.n}</b>
            <span>{s.label}</span>
          </div>
        ))}
      </div>

      <div className="grid split">
        <section>
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
            <h2 style={{ fontSize: 22 }}>Recommended timetables</h2>
            <span className="tiny">
              {front.length} non-dominated of {solutions.length} feasible
            </span>
          </div>
          <div className="grid grid-2">
            {(Object.keys(pickLabels) as PickKind[]).map((k) => {
              const s = picks[k];
              const on = s.id === selectedId;
              return (
                <div key={k} className="card pick" style={on ? { boxShadow: '0 0 0 2px var(--ink), var(--shadow)' } : undefined}>
                  <div className="row" style={{ justifyContent: 'space-between' }}>
                    <h3>{pickLabels[k].title}</h3>
                    <span className="chip">{s.id}</span>
                  </div>
                  <span className="tiny" style={{ marginTop: -6 }}>
                    {pickLabels[k].blurb}
                  </span>
                  <ObjectiveBars s={s} front={front} />
                  <div className="row">
                    <button className={`btn sm ${on ? '' : 'ghost'}`} onClick={() => select(s.id)}>
                      {on ? '✓ Selected' : 'Select'}
                    </button>
                    <button
                      className="btn sm ghost"
                      onClick={() => {
                        select(s.id);
                        nav('/designer');
                      }}
                    >
                      Design it →
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="tiny" style={{ marginTop: 14 }}>
            Bars are relative to the Pareto front: shorter is better. <Link to="/pareto">Open the full front →</Link>
          </p>
        </section>

        <section className="card tape" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <h2>Preview</h2>
            <Link to="/designer" className="btn sm">
              Customise
            </Link>
          </div>
          <PosterFrame w={size.w} h={size.h} maxH={560}>
            <TimetablePoster instance={inst} solution={selected} view={view} theme={theme} decorations={decorations} blockNotes={blockNotes} />
          </PosterFrame>
        </section>
      </div>
    </>
  );
}
