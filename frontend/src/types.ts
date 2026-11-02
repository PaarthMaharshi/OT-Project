// Data shapes mirror the proposal (Section 3/4) so the Python backend can fill them later.

export type EventType = 'theory' | 'lab' | 'tutorial';
export type RoomType = 'classroom' | 'lab' | 'hall';

export interface Group {
  id: string;
  name: string; // e.g. "SY-Comp Div A"
  strength: number;
}

export interface Faculty {
  id: string;
  name: string;
  maxWeekly: number; // H_f
  maxDaily: number;
}

export interface Room {
  id: string;
  name: string;
  capacity: number;
  type: RoomType;
}

export interface Offering {
  id: string;
  subject: string; // full name
  code: string; // short label shown on blocks
  groupIds: string[];
  type: EventType;
  sessionsPerWeek: number;
  eligibleFaculty: string[];
}

export interface TEvent {
  id: string;
  offeringId: string;
  length: number; // periods
  headcount: number;
}

export interface TimeGrid {
  days: string[];
  periods: string[]; // labels like "9:00 - 10:00"
  lunchPeriod: number; // index into periods
}

export interface Instance {
  name: string;
  grid: TimeGrid;
  groups: Group[];
  faculty: Faculty[];
  rooms: Room[];
  offerings: Offering[];
  events: TEvent[];
}

export interface Assignment {
  eventId: string;
  day: number;
  period: number; // start period
  roomId: string;
}

export interface Objectives {
  f1: number; // idle gaps
  f2: number; // room wastage (0..1)
  f3: number; // workload imbalance (std of relative load)
}

export interface Solution {
  id: string;
  objectives: Objectives;
  assignments: Assignment[];
  facultyOf: Record<string, string>; // offeringId -> facultyId
  onFront: boolean;
  violation?: number;
  metrics?: Record<string, number>; // practical metrics from the backend
}

export type Scenario = 'demo' | 'small' | 'medium' | 'large';

export interface RunParams {
  scenario: Scenario;
  instanceSeed: number;
  seed: number;
  popSize: number;
  generations: number;
  useSeeding: boolean;
  useRepair: boolean;
  useLocalSearch: boolean;
}

export interface RunStats {
  runtime: number;
  evaluations: number;
  generations: number;
  frontSize: number;
  hypervolume: number;
  baselineHypervolume: number;
  spacing: number;
  baselineFeasible: boolean;
  convergence: { generation: number; hypervolume: number; archiveSize: number; feasibleFraction: number }[];
}

export interface RunResult {
  params: Record<string, unknown>;
  instance: Instance;
  solutions: Solution[];
  baseline: Solution;
  stats: RunStats;
}

export interface RunProgress {
  generation: number;
  total: number;
  archiveSize?: number;
  feasibleFraction?: number;
  seconds?: number;
}

export type ViewKind = 'group' | 'faculty' | 'room';
