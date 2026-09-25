# -*- coding: utf-8 -*-
"""
resume_nsga_v2.py  —  Resume NSGA-II với HV tuyệt đối (Lựa chọn C)
====================================================================
Điểm khác biệt so với resume_nsga.py:

  1. HV tính TRỰC TIẾP trên không gian gốc [s/xe, kg/xe], KHÔNG chuẩn hoá.
     ref point cố định: r = [120.0 s/xe, 0.45 kg/xe]
     => Tránh hoàn toàn lỗi "Reference Point Collapse" đã phân tích.

  2. Early stopping dựa trên HV tuyệt đối với tham số được hiệu chỉnh:
       patience            = max(8, n_gen_extra // 5)  (tối thiểu 8 gen)
       min_delta           = 0.001  (0.1% thay vì 0.5%)
       min_warmup_gen  = min(15, n_gen_extra // 4)

  3. Cột HV_gen_abs (đơn vị s·kg/xe²) được ghi vào CSV ngay trong vòng lặp.

  4. Gen đánh số tiếp từ gen_old_max+1, merge full CSV tự động sau khi xong.

  5. Warm-start từ Pareto front (Rank=0) của CSV cũ + random fill phần còn lại.

Usage:
  python resume_nsga_v2.py --scenario S1 --csv_old results_S1_..._gen16.csv --n_gen_extra 60
  python resume_nsga_v2.py --scenario S2 --csv_old results_S2_..._gen16.csv --n_gen_extra 60
  python resume_nsga_v2.py --scenario S3 --csv_old results_S3_..._gen30.csv --n_gen_extra 40
"""

import os
import signal
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.indicators.hv import HV

from config import SCENARIOS
from sumo_evaluator_ver2 import SUMOEvaluatorImproved
from repair_operator_ver2 import TrafficSignalRepairImproved
from problem_v2_honest import ProblemV2_HonestObjective

# ─────────────────────────────────────────────────────────────────────────────
# REF POINT — nhất quán với compute_hv_v2.py
# ─────────────────────────────────────────────────────────────────────────────
REF_POINT_ABS  = np.array([120.0, 0.45])   # [s/xe, kg/xe]
HV_MAX_ABS     = float(REF_POINT_ABS[0] * REF_POINT_ABS[1])   # = 54.0
INVALID_THRESH = 90_000.0


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING — dùng HV tuyệt đối, không cần normalizer
# ─────────────────────────────────────────────────────────────────────────────

class AbsoluteHVEarlyStopping:
    """
    Early stopping dựa trên HV tuyệt đối (không chuẩn hoá).

    Tính HV trực tiếp trên không gian [s/xe, kg/xe] với ref point cố định.
    Không bao giờ bị ảnh hưởng bởi nghiệm kẹt xe hay running normalizer.

    Logic dừng (2 điều kiện AND):
      (1) Đã chạy ít nhất min_warmup_gen thế hệ trong session này
          → Bảo vệ giai đoạn warm-up, không đếm no_improve trong min_warmup_gen gen đầu.
      (2) no_improve >= patience  (HV không tăng > min_delta% liên tiếp)

    Sơ đồ:
      Gen 1..min_warmup_gen : chạy tự do, không đếm no_improve, không dừng
      Gen min_warmup_gen+1.. : bắt đầu đếm, dừng khi no_improve >= patience

    Parameters
    ----------
    patience        : số gen liên tiếp không cải thiện để dừng (default: 10)
    min_delta       : ngưỡng cải thiện tương đối (0.001 = 0.1%)
    min_warmup_gen  : số gen warm-up tối thiểu trước khi bắt đầu đếm (default: 15)
    ref_point       : ref point tuyệt đối [s/xe, kg/xe]
    """

    def __init__(self,
                 patience: int         = 10,
                 min_delta: float      = 0.001,
                 min_warmup_gen: int   = 15,
                 ref_point: np.ndarray = None):
        self.patience       = patience
        self.min_delta      = min_delta
        self.min_warmup_gen = min_warmup_gen
        ref                 = ref_point if ref_point is not None else REF_POINT_ABS
        self.hv_calc        = HV(ref_point=ref)

        self.best_hv        = -np.inf
        self.no_improve     = 0
        self.hv_history     = []
        self.gen_count      = 0   # đếm gen trong session resume

    def update(self, F_raw: np.ndarray) -> dict:
        """
        F_raw : Pareto front thế hệ hiện tại, NOT chuẩn hoá, shape (k, 2).
                Đơn vị: [s/xe, kg/xe].
        """
        self.gen_count += 1

        # Lọc nghiệm không dominated bởi ref point
        F_valid = F_raw[np.all(F_raw < REF_POINT_ABS, axis=1)]

        if len(F_valid) == 0:
            hv_val = max(0.0, self.best_hv if self.best_hv != -np.inf else 0.0)
            self.hv_history.append(hv_val)
            # Chỉ đếm no_improve sau warm-up
            if self.gen_count > self.min_warmup_gen:
                self.no_improve += 1
            return self._build_result(hv_val, False)

        current_hv = float(self.hv_calc.do(F_valid))
        self.hv_history.append(current_hv)

        if self.best_hv == -np.inf:
            improved = True
        else:
            improved = (current_hv - self.best_hv) / (abs(self.best_hv) + 1e-12) > self.min_delta

        if improved:
            self.best_hv    = current_hv
            self.no_improve = 0
        else:
            # Chỉ đếm no_improve sau warm-up
            if self.gen_count > self.min_warmup_gen:
                self.no_improve += 1

        return self._build_result(current_hv, improved)

    def _build_result(self, hv_val: float, improved: bool) -> dict:
        # Dừng khi ĐÃ QUA warm-up VÀ no_improve đủ patience
        in_warmup   = self.gen_count <= self.min_warmup_gen
        should_stop = (not in_warmup) and (self.no_improve >= self.patience)
        return {
            "hv"         : hv_val,
            "hv_pct"     : hv_val / HV_MAX_ABS * 100,
            "improved"   : improved,
            "should_stop": should_stop,
            "no_improve" : self.no_improve,
            "in_warmup"  : in_warmup,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

class WarmStartSampling(Sampling):
    """Khởi tạo từ seed_X (Pareto cũ) + random fill."""

    def __init__(self, seed_X: np.ndarray, repair_op, pop_size: int):
        super().__init__()
        self.seed_X   = seed_X
        self.repair   = repair_op
        self.pop_size = pop_size

    def _do(self, problem, n_samples, **kwargs):
        n_seed = min(len(self.seed_X), n_samples)
        result = list(self.seed_X[:n_seed])
        seen   = set(tuple(int(round(v)) for v in x) for x in result)
        needed = n_samples - n_seed

        attempts = 0
        while len(result) - n_seed < needed and attempts < needed * 20:
            x = np.random.randint(
                problem.xl.astype(int), problem.xu.astype(int) + 1,
                (1, problem.n_var)
            ).astype(float)
            x   = self.repair._do(problem, x, **kwargs)
            key = tuple(int(round(v)) for v in x[0])
            if key not in seen:
                seen.add(key)
                result.append(x[0])
            attempts += 1

        while len(result) < n_samples:
            x = np.random.randint(
                problem.xl.astype(int), problem.xu.astype(int) + 1,
                (1, problem.n_var)
            ).astype(float)
            x = self.repair._do(problem, x, **kwargs)
            result.append(x[0])

        return np.array(result[:n_samples])


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II với diversity infill (giữ nguyên từ run_nsga2_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

class NSGA2WithDiversityInfill(NSGA2):
    def __init__(self, *args, max_retries: int = 5, repair=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries    = max_retries
        self.repair         = repair
        self._dup_replaced  = 0
        self._random_filled = 0

    def _infill(self):
        off   = super()._infill()
        pop_X = self.pop.get("X")

        def _to_key(x):
            return tuple(int(round(v)) for v in x)

        known         = set(_to_key(row) for row in pop_X)
        off_X         = off.get("X")
        new_X, accepted_keys = [], set()

        for i in range(len(off_X)):
            key = _to_key(off_X[i])
            if key not in known and key not in accepted_keys:
                new_X.append(off_X[i].copy()); accepted_keys.add(key)
            else:
                found = False
                for attempt in range(self.max_retries):
                    parent = pop_X[np.random.randint(len(pop_X))].copy().reshape(1, -1)
                    eta_m  = max(1, 20 - attempt * 3)
                    _rep   = getattr(self.mating.mutation, "repair", None)
                    mutated = PM(prob=1.0, eta=eta_m, repair=_rep, vtype=float)._do(
                        self.problem, parent, algorithm=self)
                    nk = _to_key(mutated[0])
                    if nk not in known and nk not in accepted_keys:
                        new_X.append(mutated[0].copy()); accepted_keys.add(nk)
                        self._dup_replaced += 1; found = True; break

                if not found:
                    for _ in range(20):
                        rx = np.random.randint(
                            self.problem.xl.astype(int),
                            self.problem.xu.astype(int) + 1
                        ).astype(float)
                        if self.repair:
                            rx = self.repair._do(self.problem, rx.reshape(1,-1))[0]
                        rk = _to_key(rx)
                        if rk not in known and rk not in accepted_keys:
                            new_X.append(rx); accepted_keys.add(rk)
                            self._random_filled += 1; found = True; break
                    if not found:
                        new_X.append(off_X[i].copy())

        return Population.new("X", np.array(new_X))

    def log_diversity_stats(self):
        print(f"[DIVERSITY] Mutation replaced: {self._dup_replaced} | "
              f"Random filled: {self._random_filled}")


# ─────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

def load_seed(csv_path: str) -> tuple:
    """Trả về (seed_X, gen_old_max, df_old)."""
    df        = pd.read_csv(csv_path)
    last_gen  = int(df["Gen"].max())
    pareto    = df[(df["Gen"] == last_gen) & (df["Rank"] == 0)]
    cols      = ["C", "g1_1", "g1_2", "g2_1", "g2_2", "Offset"]
    seed_X    = pareto[cols].values.astype(float)
    print(f"[WARM-START] {len(seed_X)} nghiệm Pareto (gen={last_gen}) "
          f"từ {os.path.basename(csv_path)}")
    return seed_X, last_gen, df


def record_gen(algorithm, history_list, hv_info: dict,
               scenario_id: str, gen_offset: int, hv_by_gen: dict):
    """Ghi dữ liệu thế hệ + HV vào history và hv_by_gen."""
    pop         = algorithm.pop
    X_arr       = pop.get("X")
    F_arr       = pop.get("F")
    rank_arr    = pop.get("rank")
    crowding_arr= pop.get("crowding")
    actual_gen  = gen_offset + algorithm.n_gen

    hv_by_gen[actual_gen] = hv_info["hv"]

    for i in range(len(X_arr)):
        f1, f2 = F_arr[i]
        try:    r  = int(rank_arr[i])    if rank_arr     is not None else -1
        except: r  = -1
        try:    cr = float(crowding_arr[i]) if crowding_arr is not None else 0.0
        except: cr = 0.0
        history_list.append([
            actual_gen, *map(int, X_arr[i]),
            float(f1), float(f2), r, cr, scenario_id
        ])

    mark   = "✓" if hv_info["improved"] else "·"
    status = "[WARMUP]" if hv_info.get("in_warmup") else f"no_imp={hv_info['no_improve']:>2}"
    print(f"Gen {actual_gen:>3} (+{algorithm.n_gen:>2}) | "
          f"HV={hv_info['hv']:>8.4f} ({hv_info['hv_pct']:>5.2f}%) {mark} | {status}")


def plot_resume(df_full, hv_by_gen, hv_old_series,
                scenario_id, label, out_dir, gen_old_max):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"NSGA-II Resume (HV tuyệt đối) | {scenario_id} ({label})",
                 fontsize=13, fontweight="bold")

    last_gen = df_full["Gen"].max()

    # Subplot 1: Pareto front cuối
    ax1 = axes[0]
    lp  = df_full[df_full["Gen"] == last_gen]
    ax1.scatter(lp[lp["Rank"]>0]["f1_mean_timeloss_s_per_veh"],
                lp[lp["Rank"]>0]["f2_co2_kg_per_veh"],
                c="lightgray", s=15, label="Dominated", alpha=0.6)
    par = lp[lp["Rank"]==0]
    ax1.scatter(par["f1_mean_timeloss_s_per_veh"], par["f2_co2_kg_per_veh"],
                c="red", s=50, label=f"Pareto (n={len(par)})", zorder=3)
    # Ref point
    ax1.axvline(x=REF_POINT_ABS[0], color="purple", linestyle="--",
                linewidth=1, alpha=0.5, label=f"ref_f1={REF_POINT_ABS[0]}")
    ax1.axhline(y=REF_POINT_ABS[1], color="orange", linestyle="--",
                linewidth=1, alpha=0.5, label=f"ref_f2={REF_POINT_ABS[1]}")
    ax1.set_xlabel("f1 — Timeloss (s/xe)"); ax1.set_ylabel("f2 — CO₂ (kg/xe)")
    ax1.set_title("Không gian mục tiêu (gen cuối)")
    ax1.legend(fontsize=8); ax1.grid(True, linestyle="--", alpha=0.4)

    # Subplot 2: HV evolution
    ax2 = axes[1]
    if hv_old_series is not None and len(hv_old_series) > 0:
        ax2.plot(hv_old_series.index, hv_old_series.values,
                 marker="o", markersize=3, lw=1.5, color="steelblue",
                 label="Run cũ", alpha=0.7)
    new_gens = sorted(g for g in hv_by_gen if g > gen_old_max)
    if new_gens:
        ax2.plot(new_gens, [hv_by_gen[g] for g in new_gens],
                 marker="s", markersize=4, lw=1.8, color="tomato", label="Resume")
    ax2.axvline(x=gen_old_max, color="gray", linestyle="--", lw=1,
                label=f"Nối tại gen {gen_old_max}")
    ax2.axhline(y=HV_MAX_ABS * 0.9, color="green", linestyle=":", lw=1,
                label=f"90% HV_max={HV_MAX_ABS*0.9:.1f}")
    ax2.set_xlabel("Thế hệ"); ax2.set_ylabel(f"HV tuyệt đối (s·kg/xe²)")
    ax2.set_title(f"HV Evolution (ref={REF_POINT_ABS.tolist()})")
    ax2.legend(fontsize=8); ax2.grid(True, linestyle="--", alpha=0.4)

    # Subplot 3: Pareto size per gen
    ax3 = axes[2]
    ps     = df_full[df_full["Rank"]==0].groupby("Gen").size()
    colors = ["steelblue" if g <= gen_old_max else "tomato" for g in ps.index]
    ax3.bar(ps.index, ps.values, color=colors, edgecolor="darkred", lw=0.5)
    ax3.set_xlabel("Thế hệ"); ax3.set_ylabel("Số nghiệm Rank=0")
    ax3.set_title("Pareto size (xanh=cũ, đỏ=resume)")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"resume_{scenario_id}_{label}_gen{last_gen}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Biểu đồ: {path}")


def save_checkpoint(history_data, hv_by_gen, df_old, scenario_id,
                    scen_label, gen_old_max, results_dir, algorithm_n_gen):
    """
    Lưu CSV ngay lập tức — gọi từ signal handler (Ctrl+C) hoặc cuối vòng lặp.
    An toàn khi gọi nhiều lần (overwrite).
    """
    if not history_data:
        print("[CHECKPOINT] Không có dữ liệu để lưu.")
        return

    cols = ["Gen","C","g1_1","g1_2","g2_1","g2_2","Offset",
            "f1_mean_timeloss_s_per_veh","f2_co2_kg_per_veh",
            "Rank","CrowdingDistance","Scenario"]
    df_resume             = pd.DataFrame(history_data, columns=cols)
    df_resume["HV_gen_abs"] = df_resume["Gen"].map(hv_by_gen).fillna(0.0)

    os.makedirs(results_dir, exist_ok=True)
    gen_final = gen_old_max + algorithm_n_gen

    csv_resume = os.path.join(results_dir,
        f"resume_{scenario_id}_{scen_label}_gen{gen_old_max}to{gen_final}.csv")
    df_resume.to_csv(csv_resume, index=False)
    print(f"\n[CHECKPOINT SAVED] {csv_resume}  ({len(df_resume)} dòng)")

    # Merge full
    df_old_copy = df_old.copy()
    if "HV_gen_abs" not in df_old_copy.columns:
        df_old_copy["HV_gen_abs"] = np.nan
    df_full  = pd.concat([df_old_copy, df_resume], ignore_index=True)
    csv_full = os.path.join(results_dir,
        f"full_{scenario_id}_{scen_label}_gen{gen_final}.csv")
    df_full.to_csv(csv_full, index=False)
    print(f"[CHECKPOINT SAVED] {csv_full}  ({len(df_full)} dòng tổng)")
    return df_resume, df_full


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",    type=str,  required=True, choices=["S1","S2","S3"])
    parser.add_argument("--csv_old",     type=str,  required=True)
    parser.add_argument("--n_gen_extra", type=int,  default=20)
    parser.add_argument("--pop_size",    type=int,  default=60)
    parser.add_argument("--patience",    type=int,  default=10)
    parser.add_argument("--seed",        type=int,  default=123)
    args = parser.parse_args()

    patience = args.patience if args.patience is not None else max(8, args.n_gen_extra // 5)
    scen     = SCENARIOS[args.scenario]

    print(f"\n{'='*65}")
    print(f"RESUME NSGA-II (HV tuyệt đối) | {args.scenario} ({scen['label']})")
    print(f"ref point: {REF_POINT_ABS.tolist()} | HV_max = {HV_MAX_ABS:.2f}")
    print(f"n_gen_extra={args.n_gen_extra} | patience={patience} | min_delta=0.1%")
    print(f"{'='*65}\n")

    if not os.path.exists(args.csv_old):
        raise FileNotFoundError(f"Không tìm thấy: {args.csv_old}")

    seed_X, gen_old_max, df_old = load_seed(args.csv_old)

    # HV history cũ (nếu đã có cột HV_gen_abs từ compute_hv_v2.py)
    hv_old_series = None
    if "HV_gen_abs" in df_old.columns:
        hv_old_series = df_old.groupby("Gen")["HV_gen_abs"].first().sort_index()
        print(f"[INFO] Đọc HV cũ từ cột HV_gen_abs ({len(hv_old_series)} gen)")
    else:
        print("[INFO] CSV cũ chưa có cột HV_gen_abs. Chạy compute_hv_v2.py trước "
              "để có đường HV đầy đủ trên biểu đồ.")

    # Khởi tạo
    sumo_runner = SUMOEvaluatorImproved(
        sumo_config_path=scen["sumo_cfg"],
        total_hourly_demand=scen["total_hourly_demand"],
        scenario_id=args.scenario
    )
    problem   = ProblemV2_HonestObjective(evaluator=sumo_runner)
    repair_v2 = TrafficSignalRepairImproved()

    algorithm = NSGA2WithDiversityInfill(
        pop_size            = args.pop_size,
        max_retries         = 5,
        repair              = repair_v2,
        sampling            = WarmStartSampling(seed_X, repair_v2, args.pop_size),
        crossover           = SBX(prob=0.9, eta=15, repair=repair_v2, vtype=float),
        mutation            = PM(prob=0.2, eta=20, repair=repair_v2, vtype=float),
        eliminate_duplicates= True,
    )

    early_stop = AbsoluteHVEarlyStopping(
        patience            = patience,
        min_delta           = 0.001,
        min_warmup_gen = min(15, args.n_gen_extra // 4),
        ref_point           = REF_POINT_ABS,
    )

    algorithm.setup(problem, termination=("n_gen", args.n_gen_extra), seed=args.seed)

    history_data = []
    hv_by_gen    = {}
    results_dir  = os.path.join(os.getcwd(), "results")

    # ── Signal handler: Ctrl+C lưu dữ liệu rồi thoát ───────────────────────
    _interrupted = [False]   # dùng list để closure có thể gán

    def _handle_sigint(sig, frame):
        if _interrupted[0]:
            print("\n[FORCE EXIT] Ctrl+C lần 2 — thoát ngay không lưu.")
            os._exit(1)
        _interrupted[0] = True
        print("\n[Ctrl+C] Nhận tín hiệu dừng — đang lưu dữ liệu...")
        save_checkpoint(history_data, hv_by_gen, df_old,
                        args.scenario, scen["label"],
                        gen_old_max, results_dir, algorithm.n_gen)
        print("[Ctrl+C] Đã lưu xong. Thoát.")
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)
    print("[INFO] Nhấn Ctrl+C một lần để dừng và lưu dữ liệu an toàn.\n")

    # ── Vòng lặp chính ──────────────────────────────────────────────────────
    while algorithm.has_next():
        algorithm.next()
        pop      = algorithm.pop
        F        = pop.get("F")
        rank     = pop.get("rank")
        feasible = pop.get("feasible")

        if feasible is not None:
            mask_valid = feasible.flatten()
        else:
            mask_valid = np.all(F < INVALID_THRESH, axis=1)

        if rank is not None:
            rank_safe  = np.array([r if r is not None else 999
                                   for r in rank.flatten()], dtype=int)
            mask_front = (rank_safe == 0) & mask_valid
        else:
            mask_front = np.zeros(len(F), dtype=bool)

        F_front = F[mask_front]

        if len(F_front) > 0:
            hv_info = early_stop.update(F_front)
        else:
            early_stop.gen_count  += 1
            hv_val = max(0.0, early_stop.best_hv if early_stop.best_hv != -np.inf else 0.0)
            early_stop.hv_history.append(hv_val)
            hv_info = early_stop._build_result(hv_val, False)

        record_gen(algorithm, history_data, hv_info,
                   args.scenario, gen_old_max, hv_by_gen)

        if algorithm.n_gen % 5 == 0:
            algorithm.log_diversity_stats()
            # Auto-checkpoint mỗi 5 gen — an toàn nếu mất điện/crash
            save_checkpoint(history_data, hv_by_gen, df_old,
                            args.scenario, scen["label"],
                            gen_old_max, results_dir, algorithm.n_gen)

        if hv_info["should_stop"]:
            print(f"\n[DỪNG SỚM] HV không cải thiện > 0.1% trong {patience} gen (sau warm-up {early_stop.min_warmup_gen} gen)")
            break

    algorithm.log_diversity_stats()

    # ── Lưu CSV lần cuối + vẽ biểu đồ ──────────────────────────────────────
    result = save_checkpoint(history_data, hv_by_gen, df_old,
                             args.scenario, scen["label"],
                             gen_old_max, results_dir, algorithm.n_gen)
    if result:
        df_resume, df_full = result
        # Vẽ biểu đồ chỉ khi hoàn thành bình thường (không phải Ctrl+C)
        plot_resume(df_full, hv_by_gen, hv_old_series,
                    args.scenario, scen["label"], results_dir, gen_old_max)

    # Tóm tắt
    final_pareto = df_resume[(df_resume["Gen"]==df_resume["Gen"].max()) &
                             (df_resume["Rank"]==0)]
    print(f"\n{'='*65}")
    print(f"HOÀN THÀNH | Gen: {gen_old_max} → {gen_final}")
    print(f"Pareto front cuối: {len(final_pareto)} nghiệm")
    print(f"HV tốt nhất (resume): {early_stop.best_hv:.4f} "
          f"({early_stop.best_hv/HV_MAX_ABS*100:.2f}% HV_max)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()