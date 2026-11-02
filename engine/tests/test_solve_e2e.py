"""End-to-end: frontend-format JSON -> solve() -> schema check + independent zero-violation check."""
import copy
import json
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

import adapter
from engine import solve
from engine.frontend_bridge import from_frontend
from engine.generator import generate
from engine.problem import TimetableProblem

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SAMPLE = json.loads((ROOT / "samples" / "frontend_instance.json").read_text())
LAB = generate(1)
FAST = {"pop_size": 30, "evaluations": 900, "seed": 1}


def rebuild_and_check(engine_instance: dict, objectives: list, entry: dict) -> float:
    """Independent check: turn the returned timetable back into a solution vector and re-evaluate it."""
    p = TimetableProblem(engine_instance, objectives)
    lec_index = {l: i for i, l in enumerate(p.lecture_ids)}
    room_index = {r: i for i, r in enumerate(p.room_ids)}
    x = np.zeros(2 * p.S, dtype=int)
    seen = set()
    for e in entry["timetable"]:
        li = lec_index[e["lecture_id"]]
        s = int(np.flatnonzero((p.sess_lecture == li) & (p.sess_number == e["session"]))[0])
        seen.add(s)
        x[s] = e["day"] * p.P + e["slot"]
        x[p.S + s] = room_index[e["room_id"]]
    assert len(seen) == p.S, "every session must appear exactly once"
    F, V = p.evaluate(x)
    np.testing.assert_allclose(F, entry["objectives"])
    return V


class TestEndToEnd(unittest.TestCase):
    def test_frontend_json_gives_valid_schema_and_zero_violations(self):
        inst = copy.deepcopy(FRONTEND_SAMPLE)
        inst["params"] = {**FAST}
        r = solve(inst)
        self.assertNotIn("error", r, r.get("details"))
        json.dumps(r)  # JSON-serialisable
        for key in ("front", "recommended", "ranking", "metrics", "history"):
            self.assertIn(key, r)
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["front"])
        for k in ("hv", "igd", "spacing", "feasible_rate", "runtime_s"):
            self.assertIn(k, r["metrics"])
        n = len(r["front"])
        self.assertTrue(0 <= r["recommended"] < n)
        self.assertEqual(sorted(r["ranking"]["topsis"]), list(range(n)))
        self.assertEqual(sorted(r["ranking"]["vikor"]), list(range(n)))
        self.assertTrue(all(set(h) >= {"gen", "hv"} for h in r["history"]))
        engine_inst, _, _ = from_frontend(inst)
        for sol in r["front"]:
            self.assertEqual(sol["violation"], 0)
            self.assertEqual(set(sol["timetable"][0]) >= {"lecture_id", "day", "slot", "room_id"}, True)
            self.assertEqual(rebuild_and_check(engine_inst, r["objective_names"], sol), 0)

    def test_lab_format_every_algorithm(self):
        for algo in ["nsga2", "constrained_nsga2", "spea2", "moead", "mosa", "moma", "nsga3",
                     "weighted_sum", "eps_constraint", "goal_programming", "lexicographic"]:
            inst = {**LAB, "algorithm": algo, "params": {**FAST, "objectives": ["student_gaps", "teacher_gaps"]}}
            r = solve(inst)
            self.assertNotIn("error", r, (algo, r.get("details")))
            self.assertIn(r["status"], ("ok", "no_feasible_solution"), algo)
            if algo != "nsga2":  # the penalty-only baseline may fail to reach feasibility on a short budget
                self.assertEqual(r["status"], "ok", algo)

    def test_determinism(self):
        inst = {**LAB, "params": {**FAST, "seed": 7}}
        a, b = solve(inst), solve(inst)
        self.assertEqual([f["objectives"] for f in a["front"]], [f["objectives"] for f in b["front"]])

    def test_time_limit_returns_best_so_far(self):
        inst = {**LAB, "params": {"pop_size": 40, "evaluations": 10_000_000, "seed": 1, "time_limit_s": 1.0}}
        t = time.perf_counter()
        r = solve(inst)
        self.assertLess(time.perf_counter() - t, 6)
        self.assertEqual(r["metrics"]["stopped"], "time_limit")
        self.assertTrue(r["front"])

    def test_progress_callback_and_cancellation(self):
        calls = []

        def progress(entry):
            calls.append(entry)
            return len(calls) < 3  # returning False on the 3rd call cancels the run

        r = solve({**LAB, "params": {**FAST, "evaluations": 100_000}}, progress=progress)
        self.assertEqual(r["metrics"]["stopped"], "cancelled")
        self.assertEqual(len(calls), 3)
        self.assertTrue({"gen", "hv", "best_violation"} <= set(calls[0]))
        flag = {"n": 0}

        def should_stop():
            flag["n"] += 1
            return flag["n"] > 2

        r = solve({**LAB, "params": {**FAST, "evaluations": 100_000}}, should_stop=should_stop)
        self.assertEqual(r["metrics"]["stopped"], "cancelled")


class TestValidationErrors(unittest.TestCase):
    def assertError(self, inst, fragment):
        r = solve(inst)
        self.assertIn("error", r)
        self.assertTrue(any(fragment in d for d in r["details"]), r["details"])

    def test_unknown_ids(self):
        bad = copy.deepcopy(LAB)
        bad["lectures"][0]["teacher_id"] = "NOPE"
        self.assertError(bad, "unknown teacher_id 'NOPE'")

    def test_group_larger_than_every_room(self):
        bad = copy.deepcopy(LAB)
        bad["groups"][0]["size"] = 9999
        self.assertError(bad, "largest room holds only")

    def test_more_sessions_than_slots(self):
        bad = copy.deepcopy(LAB)
        bad["lectures"][0]["sessions_per_week"] = 40
        self.assertError(bad, "week has only 30 slots")

    def test_unknown_algorithm_and_bad_params(self):
        self.assertError({**LAB, "algorithm": "magic"}, "Unknown algorithm")
        self.assertError({**LAB, "params": {"pop_size": -3}}, "params.pop_size")

    def test_garbage_never_crashes(self):
        for junk in [None, "text", 42, [], {}, {"days": "five"}]:
            r = solve(junk)
            self.assertIn("error", r)


class TestAdapterHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = adapter.make_server("127.0.0.1", 0)  # port 0 = any free port
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_algorithms_solve(self):
        self.assertEqual(self.call("/engine/health"), (200, {"ok": True}))
        code, algos = self.call("/engine/algorithms")
        self.assertEqual(len(algos), 11)
        code, r = self.call("/engine/solve", {**LAB, "params": FAST})
        self.assertEqual(code, 200)
        self.assertEqual(r["status"], "ok")
        code, r = self.call("/engine/solve", {**LAB, "algorithm": "magic"})
        self.assertEqual(code, 400)

    def test_background_run_and_cancel(self):
        code, r = self.call("/engine/runs", {**LAB, "params": {**FAST, "evaluations": 1_000_000}})
        run_id = r["run_id"]
        time.sleep(0.5)
        self.assertEqual(self.call(f"/engine/runs/{run_id}/cancel", {})[0], 200)
        for _ in range(100):
            code, st = self.call(f"/engine/runs/{run_id}")
            if st["status"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(st["status"], "cancelled")
        self.assertEqual(st["result"]["metrics"]["stopped"], "cancelled")
        self.assertTrue(st["result"]["front"])


if __name__ == "__main__":
    unittest.main()
