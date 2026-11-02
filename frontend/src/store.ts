import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Instance, RunParams, RunProgress, RunResult, RunStats, Solution, ViewKind } from './types';
import { mockInstance } from './mock/instance';
import { generateSolutions, representativePicks } from './mock/solutions';
import { palettes, type Palette } from './themes/palettes';
import type { Aspect, BgStyle } from './themes/designs';
import { backendAvailable, optimise } from './api';

export type ColorRef = 'c1' | 'c2' | 'c3' | 'ink' | string; // palette slot or a hex value

export interface Deco {
  id: string;
  kind: 'sticker' | 'word' | 'note' | 'callout';
  x: number; // % of poster width
  y: number; // % of poster height
  size: number; // px (sticker width / note width / text size)
  rot: number; // degrees
  sticker?: string;
  text?: string;
  color: ColorRef;
  arrowAngle?: number; // callouts
  arrowLen?: number;
}

export interface BlockNote {
  icon: string;
  text: string;
}

export interface ThemeSettings {
  palette: Palette;
  fontId: string;
  designId: string;
  bgStyle: BgStyle;
  aspect: Aspect;
  title: string;
  subtitle: string;
  showTeacher: boolean;
  showRoom: boolean;
  showTimeCol: boolean;
  showLegend: boolean;
  useCodes: boolean;
}

export type DataSource = 'demo' | 'backend';

interface State {
  // optimisation data
  instance: Instance;
  solutions: Solution[];
  baseline: Solution | null;
  stats: RunStats | null;
  source: DataSource;
  runParams: RunParams;
  running: boolean;
  progress: RunProgress | null;
  runError: string | null;
  selectedId: string;
  // design
  view: { kind: ViewKind; entityId: string };
  theme: ThemeSettings;
  savedPalettes: Palette[];
  decorations: Deco[];
  selectedDecoId: string | null;
  blockNotes: Record<string, BlockNote>;

  setRunParams: (p: Partial<RunParams>) => void;
  runOptimizer: () => Promise<void>;
  select: (id: string) => void;
  setView: (kind: ViewKind, entityId?: string) => void;
  setTheme: (p: Partial<ThemeSettings>) => void;
  savePalette: (name: string) => void;
  deletePalette: (id: string) => void;
  addDeco: (d: Omit<Deco, 'id'>) => void;
  updateDeco: (id: string, p: Partial<Deco>) => void;
  removeDeco: (id: string) => void;
  selectDeco: (id: string | null) => void;
  clearDecos: () => void;
  setBlockNote: (eventId: string, note: BlockNote | null) => void;
}

const RUN_KEY = 'clockwork-run'; // last backend result, kept apart from the small settings blob

const kneeOf = (sols: Solution[]) => {
  const front = sols.filter((s) => s.onFront);
  return front.length ? representativePicks(front).knee.id : sols[0]?.id ?? '';
};

function loadSavedRun(): RunResult | null {
  try {
    const raw = localStorage.getItem(RUN_KEY);
    return raw ? (JSON.parse(raw) as RunResult) : null;
  } catch {
    return null;
  }
}

function saveRun(r: RunResult) {
  try {
    localStorage.setItem(RUN_KEY, JSON.stringify(r));
  } catch {
    // Too big for localStorage (large instances): keep only the Pareto front.
    try {
      localStorage.setItem(RUN_KEY, JSON.stringify({ ...r, solutions: r.solutions.filter((s) => s.onFront) }));
    } catch {
      /* give up silently; the run stays in memory */
    }
  }
}

const firstEntity = (inst: Instance, kind: ViewKind) =>
  kind === 'group' ? inst.groups[0].id : kind === 'faculty' ? inst.faculty[0].id : inst.rooms[0].id;

const saved = loadSavedRun();
const demoSolutions = generateSolutions(mockInstance, 7);
const initialData = saved
  ? { instance: saved.instance, solutions: saved.solutions, baseline: saved.baseline, stats: saved.stats, source: 'backend' as DataSource }
  : { instance: mockInstance, solutions: demoSolutions, baseline: null, stats: null, source: 'demo' as DataSource };

export const defaultTheme: ThemeSettings = {
  palette: { ...palettes[0] },
  fontId: 'retro',
  designId: 'pill',
  bgStyle: 'solid',
  aspect: 'phone',
  title: 'Weekly Schedule',
  subtitle: 'Third Year, First Semester',
  showTeacher: true,
  showRoom: true,
  showTimeCol: true,
  showLegend: true,
  useCodes: true,
};

export const defaultRunParams: RunParams = {
  scenario: 'demo',
  instanceSeed: 1,
  seed: 1,
  popSize: 60,
  generations: 40,
  useSeeding: true,
  useRepair: true,
  useLocalSearch: true,
};

const uid = () => Math.random().toString(36).slice(2, 9);

export const useStore = create<State>()(
  persist(
    (set, get) => ({
      ...initialData,
      runParams: defaultRunParams,
      running: false,
      progress: null,
      runError: null,
      selectedId: kneeOf(initialData.solutions),
      view: { kind: 'group', entityId: initialData.instance.groups[0].id },
      theme: defaultTheme,
      savedPalettes: [],
      decorations: [],
      selectedDecoId: null,
      blockNotes: {},

      setRunParams: (p) => set({ runParams: { ...get().runParams, ...p } }),

      runOptimizer: async () => {
        set({ running: true, runError: null, progress: { generation: 0, total: get().runParams.generations } });
        const params = get().runParams;
        if (!(await backendAvailable())) {
          // No Python server: regenerate demo data so the UI still works.
          await new Promise((r) => setTimeout(r, 400));
          const solutions = generateSolutions(mockInstance, Math.floor(Math.random() * 1e6));
          set({
            instance: mockInstance,
            solutions,
            baseline: null,
            stats: null,
            source: 'demo',
            selectedId: kneeOf(solutions),
            view: { kind: 'group', entityId: mockInstance.groups[0].id },
            blockNotes: {},
            running: false,
            progress: null,
            runError: 'The optimiser server is not running, so this is demo data. Start it with: python -m uvicorn api.server:app --port 8000',
          });
          return;
        }
        try {
          const res = await optimise(params, (p) => set({ progress: p }));
          saveRun(res);
          const keepView = get().source === 'backend' && get().instance.name === res.instance.name;
          set({
            instance: res.instance,
            solutions: res.solutions,
            baseline: res.baseline,
            stats: res.stats,
            source: 'backend',
            selectedId: kneeOf(res.solutions),
            view: keepView ? get().view : { kind: 'group', entityId: res.instance.groups[0].id },
            blockNotes: keepView ? get().blockNotes : {},
            running: false,
            progress: null,
          });
        } catch (e) {
          set({ running: false, progress: null, runError: e instanceof Error ? e.message : String(e) });
        }
      },
      select: (id) => set({ selectedId: id }),
      setView: (kind, entityId) => set({ view: { kind, entityId: entityId ?? firstEntity(get().instance, kind) } }),
      setTheme: (p) => set({ theme: { ...get().theme, ...p } }),
      savePalette: (name) => {
        const p = { ...get().theme.palette, id: 'custom-' + uid(), name };
        set({ savedPalettes: [...get().savedPalettes, p], theme: { ...get().theme, palette: p } });
      },
      deletePalette: (id) => set({ savedPalettes: get().savedPalettes.filter((p) => p.id !== id) }),
      addDeco: (d) => {
        const id = uid();
        set({ decorations: [...get().decorations, { ...d, id }], selectedDecoId: id });
      },
      updateDeco: (id, p) => set({ decorations: get().decorations.map((d) => (d.id === id ? { ...d, ...p } : d)) }),
      removeDeco: (id) =>
        set({ decorations: get().decorations.filter((d) => d.id !== id), selectedDecoId: get().selectedDecoId === id ? null : get().selectedDecoId }),
      selectDeco: (id) => set({ selectedDecoId: id }),
      clearDecos: () => set({ decorations: [], selectedDecoId: null }),
      setBlockNote: (eventId, note) => {
        const next = { ...get().blockNotes };
        if (note && (note.text || note.icon)) next[eventId] = note;
        else delete next[eventId];
        set({ blockNotes: next });
      },
    }),
    {
      name: 'timetable-studio',
      version: 2,
      partialize: (s) => ({
        selectedId: s.selectedId,
        view: s.view,
        theme: s.theme,
        runParams: s.runParams,
        savedPalettes: s.savedPalettes,
        decorations: s.decorations,
        blockNotes: s.blockNotes,
      }),
      migrate: (old) => old as never,
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<State>;
        const inst = current.instance;
        const selectedId = current.solutions.some((s) => s.id === p.selectedId) ? p.selectedId! : current.selectedId;
        // Drop a saved view that points at an entity this instance does not have.
        const v = p.view;
        const exists =
          v &&
          (v.kind === 'group' ? inst.groups : v.kind === 'faculty' ? inst.faculty : inst.rooms).some((x) => x.id === v.entityId);
        return {
          ...current,
          ...p,
          theme: { ...defaultTheme, ...p.theme },
          runParams: { ...defaultRunParams, ...p.runParams },
          selectedId,
          view: exists ? v! : current.view,
        };
      },
    },
  ),
);

export const useSelectedSolution = () => useStore((s) => s.solutions.find((x) => x.id === s.selectedId) ?? s.solutions[0]);
