# -*- coding: utf-8 -*-
"""
nsga2_core.py

Lõi thuật toán NSGA-II — tách ra từ run_nsga2_v2.py để dùng chung giữa:
    1. CLI gốc (run_nsga2_v2.py)      — vẫn chạy python run_nsga2_v2.py như cũ
    2. Service layer (FastAPI/MCP)    — gọi run_optimization() trực tiếp

Nguyên tắc tách:
    - run_optimization() KHÔNG ghi file (side-effect free) — chỉ trả về
      OptimizationResult trong bộ nhớ. Việc ghi CSV/PNG chuyển sang
      save_results(), gọi riêng khi cần.
    - Thêm tham số on_generation (callback) để báo tiến độ ra ngoài sau
      MỖI thế hệ, thay vì chỉ print() ra console — đây là móc nối để
      FastAPI/MCP server cập nhật trạng thái job cho client polling.
    - Thêm should_cancel (callback) để job có thể bị huỷ giữa chừng
      (hữu ích khi expose qua API — người dùng có thể muốn dừng job
      đang chạy nhiều giờ).

Toàn bộ logic thuật toán (RepairedSampling, HVEarlyStopping,
RunningNormalizer, NSGA2WithDiversityInfill, plot_results) được giữ
NGUYÊN VẸN so với run_nsga2_v2.py gốc — chỉ di chuyển vị trí.
"""

# === Imports ===
# Standard library
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # QUAN TRỌNG: worker chạy nền/không có display -> phải dùng backend non-GUI
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.indicators.hv import HV

# Local (giữ nguyên các module gốc của dự án)
from config import SCENARIOS
from sumo_evaluator_ver2 import SUMOEvaluatorImproved
from repair_operator_ver2 import TrafficSignalRepairImproved
from problem_v2_honest import ProblemV2_HonestObjective


# === Dataclasses cho giao tiếp với lớp service (FastAPI / MCP) ===


@dataclass
class GenerationEvent:
    """Sự kiện tiến độ phát ra sau mỗi thế hệ NSGA-II.

    Dùng để service layer (JobManager) cập nhật trạng thái job mà
    không cần đọc log console hay poll file CSV.
    """

    scenario: str
    gen: int
    n_gen_max: int
    hv: float
    improved: bool
    pareto_size: int
    stopped: bool = False
    stop_reason: Optional[str] = None


@dataclass
class OptimizationResult:
    """Kết quả đầy đủ sau khi một lần chạy NSGA-II kết thúc."""

    scenario: str
    label: str
    history_df: pd.DataFrame
    hv_history: List[float]
    gen_final: int
    csv_path: Optional[str] = None
    plot_path: Optional[str] = None


# === Các lớp thuật toán (copy nguyên vẹn từ run_nsga2_v2.py) ===


class RepairedSampling(Sampling):
    def __init__(self, repair_operator):
        super().__init__()
        self.repair = repair_operator

    def _do(self, problem, n_samples, **kwargs):
        """Sinh n_samples nghiệm khả thi unique ngay từ đầu."""
        seen = set()
        result = []
        max_attempts = n_samples * 10
        attempts = 0
        while len(result) < n_samples and attempts < max_attempts:
            x = np.random.randint(
                problem.xl.astype(int),
                problem.xu.astype(int) + 1,
                (1, problem.n_var),
            ).astype(float)
            x = self.repair._do(problem, x, **kwargs)
            key = tuple(int(round(v)) for v in x[0])
            if key not in seen:
                seen.add(key)
                result.append(x[0])
            attempts += 1
        while len(result) < n_samples:
            x = np.random.randint(
                problem.xl.astype(int),
                problem.xu.astype(int) + 1,
                (1, problem.n_var),
            ).astype(float)
            x = self.repair._do(problem, x, **kwargs)
            result.append(x[0])
        return np.array(result)


class HVEarlyStopping:
    """Early stopping dựa trên Hypervolume với hệ quy chiếu động."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        ref_point: Optional[np.ndarray] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.ref_point = ref_point if ref_point is not None else np.array([1.1, 1.1])
        self.hv_calc = HV(ref_point=self.ref_point)
        self.best_hv = -np.inf
        self.best_F_raw = None
        self.no_improve = 0
        self.hv_history: List[float] = []

    def update(self, F_raw: np.ndarray, normalizer) -> dict:
        F_norm_curr = normalizer.normalize(F_raw)
        F_norm_curr = F_norm_curr[np.all(F_norm_curr <= 1.05, axis=1)]

        if len(F_norm_curr) == 0:
            self.no_improve += 1
            return {
                "hv": max(0.0, self.best_hv if self.best_hv != -np.inf else 0.0),
                "improved": False,
                "should_stop": self.no_improve >= self.patience,
                "no_improve": self.no_improve,
            }

        current_hv = self.hv_calc.do(F_norm_curr)
        self.hv_history.append(current_hv)

        if self.best_F_raw is None:
            improved = True
        else:
            F_norm_best = normalizer.normalize(self.best_F_raw)
            F_norm_best = F_norm_best[np.all(F_norm_best <= 1.05, axis=1)]
            adjusted_best_hv = self.hv_calc.do(F_norm_best) if len(F_norm_best) > 0 else 0.0
            improved = (current_hv - adjusted_best_hv) / (abs(adjusted_best_hv) + 1e-12) > self.min_delta

        if improved:
            self.best_hv = current_hv
            self.best_F_raw = F_raw.copy()
            self.no_improve = 0
        else:
            self.no_improve += 1

        return {
            "hv": current_hv,
            "improved": improved,
            "should_stop": self.no_improve >= self.patience,
            "no_improve": self.no_improve,
        }


class RunningNormalizer:
    def __init__(self, n_obj: int = 2):
        self.f_min, self.f_max = np.full(n_obj, np.inf), np.full(n_obj, -np.inf)

    def update(self, F: np.ndarray):
        self.f_min = np.minimum(self.f_min, F.min(axis=0))
        self.f_max = np.maximum(self.f_max, F.max(axis=0))

    def normalize(self, F: np.ndarray) -> np.ndarray:
        denom = np.where(self.f_max - self.f_min < 1e-12, 1.0, self.f_max - self.f_min)
        return (F - self.f_min) / denom


def record_and_print_gen(algorithm, history_list, hv_info: dict, scenario_id: str):
    pop = algorithm.pop
    X_arr = pop.get("X")
    F_arr = pop.get("F")
    rank_arr = pop.get("rank")
    crowding_arr = pop.get("crowding")

    for i, x in enumerate(X_arr):
        f1, f2 = F_arr[i]
        try:
            r = int(rank_arr[i]) if rank_arr is not None else -1
        except (TypeError, ValueError, IndexError):
            r = -1
        try:
            cr = float(crowding_arr[i]) if crowding_arr is not None else 0.0
        except (TypeError, ValueError, IndexError):
            cr = 0.0
        history_list.append([algorithm.n_gen, *map(int, x), float(f1), float(f2), r, cr, scenario_id])

    print(
        f"Hoàn thành Thế hệ {algorithm.n_gen:^2} | HV={hv_info['hv']:.6f} | "
        f"Cải thiện: {'CÓ' if hv_info['improved'] else 'KHÔNG'}"
    )


class NSGA2WithDiversityInfill(NSGA2):
    """NSGA-II với cơ chế retry sinh offspring để chống trùng lặp."""

    def __init__(self, *args, max_retries: int = 5, repair=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.repair = repair
        self._dup_replaced = 0
        self._random_filled = 0

    def _infill(self):
        off = super()._infill()
        pop_X = self.pop.get("X")

        def _to_key(x):
            return tuple(int(round(v)) for v in x)

        known = set(_to_key(row) for row in pop_X)
        off_X = off.get("X")
        n = len(off_X)
        new_X = []
        accepted_keys = set()

        for i in range(n):
            key = _to_key(off_X[i])
            if key not in known and key not in accepted_keys:
                new_X.append(off_X[i].copy())
                accepted_keys.add(key)
            else:
                found = False
                for attempt in range(self.max_retries):
                    parent_idx = np.random.randint(len(pop_X))
                    candidate = pop_X[parent_idx].copy().reshape(1, -1)
                    eta_m = max(1, 20 - attempt * 3)
                    _repair_op = getattr(self.mating.mutation, "repair", None)
                    pm_strong = PM(prob=1.0, eta=eta_m, repair=_repair_op, vtype=float)
                    mutated = pm_strong._do(self.problem, candidate, algorithm=self)
                    new_key = _to_key(mutated[0])

                    if new_key not in known and new_key not in accepted_keys:
                        new_X.append(mutated[0].copy())
                        accepted_keys.add(new_key)
                        self._dup_replaced += 1
                        found = True
                        break

                if not found:
                    for _ in range(20):
                        rand_x = np.random.randint(
                            self.problem.xl.astype(int),
                            self.problem.xu.astype(int) + 1,
                        ).astype(float)
                        if self.repair is not None:
                            rand_x = self.repair._do(self.problem, rand_x.reshape(1, -1))[0]
                        rk = _to_key(rand_x)
                        if rk not in known and rk not in accepted_keys:
                            new_X.append(rand_x)
                            accepted_keys.add(rk)
                            self._random_filled += 1
                            found = True
                            break
                    if not found:
                        new_X.append(off_X[i].copy())

        new_off = Population.new("X", np.array(new_X))
        return new_off

    def log_diversity_stats(self):
        print(
            f"[DIVERSITY] Trùng lặp thay thế bằng mutation: {self._dup_replaced} | "
            f"Thay bằng random: {self._random_filled}"
        )


def plot_results(
    df: pd.DataFrame,
    scenario_id: str,
    label: str,
    out_dir: str,
    hv_history: list,
    gen_final: int,
) -> str:
    """Vẽ 3 biểu đồ tổng kết. Trả về đường dẫn file PNG đã lưu."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"NSGA-II | Kịch bản {scenario_id} ({label}) | Gen kết thúc: {gen_final}",
        fontsize=13,
        fontweight="bold",
    )

    ax1 = axes[0]
    col_f1 = "f1_mean_timeloss_s_per_veh"
    col_f2 = "f2_co2_kg_per_veh"

    mask_error = df["Rank"] == -1
    mask_dominated = df["Rank"] > 0
    mask_pareto = df["Rank"] == 0

    if mask_dominated.any():
        ax1.scatter(
            df.loc[mask_dominated, col_f1], df.loc[mask_dominated, col_f2],
            c="lightgray", s=15, label="Dominated", zorder=1, alpha=0.6,
        )
    if mask_pareto.any():
        ax1.scatter(
            df.loc[mask_pareto, col_f1], df.loc[mask_pareto, col_f2],
            c="red", s=40, label="Pareto (Rank=0)", zorder=3,
        )
    if mask_error.any():
        ax1.scatter(
            df.loc[mask_error, col_f1], df.loc[mask_error, col_f2],
            c="gold", marker="x", s=50, label="Lỗi (Rank=-1)", zorder=2,
        )

    ax1.set_xlabel("f1 — Control Delay (s/xe)", fontsize=10)
    ax1.set_ylabel("f2 — CO₂ (kg/xe)", fontsize=10)
    ax1.set_title("Không gian mục tiêu (f1 vs f2)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = axes[1]
    if hv_history:
        gens = list(range(1, len(hv_history) + 1))
        ax2.plot(gens, hv_history, marker="o", markersize=4, linewidth=1.5, color="steelblue")
        ax2.set_xlabel("Thế hệ", fontsize=10)
        ax2.set_ylabel("Hypervolume (chuẩn hóa)", fontsize=10)
        ax2.set_title("Lịch sử Hypervolume", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "Không có dữ liệu HV", ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_title("Lịch sử Hypervolume", fontsize=11)

    ax3 = axes[2]
    pareto_size_per_gen = (
        df[df["Rank"] == 0].groupby("Gen").size().reindex(range(1, gen_final + 1), fill_value=0)
    )
    if not pareto_size_per_gen.empty:
        ax3.bar(pareto_size_per_gen.index, pareto_size_per_gen.values, color="tomato", edgecolor="darkred", linewidth=0.5)
        ax3.set_xlabel("Thế hệ", fontsize=10)
        ax3.set_ylabel("Số nghiệm Rank=0", fontsize=10)
        ax3.set_title("Kích thước Pareto front theo thế hệ", fontsize=11)
        ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        ax3.text(0.5, 0.5, "Không có dữ liệu Rank=0", ha="center", va="center", transform=ax3.transAxes, fontsize=11)
        ax3.set_title("Kích thước Pareto front theo thế hệ", fontsize=11)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"results_{scenario_id}_{label}_gen{gen_final}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ: {plot_path}")
    return plot_path


# === Hàm lõi tái sử dụng được (điểm nối chính với FastAPI/MCP) ===


def run_optimization(
    scenario: str,
    pop_size: int = 60,
    n_gen_max: int = 30,
    seed: int = 42,
    total_hourly_demand: Optional[float] = None,
    cycle_bounds: Optional[Tuple[float, float]] = None,
    green_bounds: Optional[Tuple[float, float]] = None,
    on_generation: Optional[Callable[[GenerationEvent], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> OptimizationResult:
    """Chạy NSGA-II cho một kịch bản, KHÔNG ghi file ra đĩa.

    Đây là hàm DUY NHẤT mà cả CLI (run_nsga2_v2.py) lẫn service layer
    (FastAPI/MCP) nên gọi — tránh trùng lặp logic vòng lặp tối ưu.

    Args:
        scenario: Khóa kịch bản, "S1" | "S2" | "S3" (xem config.SCENARIOS).
        pop_size: Kích thước quần thể NSGA-II.
        n_gen_max: Số thế hệ tối đa trước khi dừng (có thể dừng sớm hơn).
        seed: Seed ngẫu nhiên cho pymoo — cố định để tái lập kết quả.
        total_hourly_demand: Ghi đè lưu lượng xe/giờ của kịch bản (mặc định
            lấy từ config.SCENARIOS[scenario] nếu không truyền). Tham số
            CÓ THỂ chỉnh — không cần model SUMO mới.
        cycle_bounds: Ghi đè (min, max) chu kỳ đèn chung C [giây] (mặc định
            (60, 120), giống hệt bounds gốc nếu không truyền).
        green_bounds: Ghi đè (min, max) thời gian xanh mỗi pha [giây] (mặc
            định (15, 90), giống hệt bounds gốc nếu không truyền).
        on_generation: Callback gọi sau MỖI thế hệ với một GenerationEvent.
            Dùng để báo tiến độ real-time cho client (ví dụ JobManager
            trong service layer cập nhật trạng thái job).
            LƯU Ý: callback chạy TRONG THREAD/PROCESS thực thi
            run_optimization — nếu cập nhật state dùng chung (dict/DB),
            phải đảm bảo thread-safe (xem core/jobs.py trong service).
        should_cancel: Callback trả True để yêu cầu dừng job giữa chừng
            (ví dụ người dùng gọi API huỷ job). Kiểm tra trước mỗi thế hệ.

    Returns:
        OptimizationResult chứa lịch sử đầy đủ (DataFrame), hv_history và
        thế hệ dừng cuối cùng. Chưa có csv_path/plot_path — gọi
        save_results() nếu cần ghi ra đĩa.

    Raises:
        KeyError: Nếu `scenario` không có trong config.SCENARIOS.
        RuntimeError: Nếu SUMO crash liên tiếp quá `max_retries` lần
            (lan truyền từ SUMOEvaluatorImproved).
    """
    if scenario not in SCENARIOS:
        raise KeyError(f"Kịch bản '{scenario}' không tồn tại. Có: {list(SCENARIOS)}")

    scen = SCENARIOS[scenario]
    demand = total_hourly_demand if total_hourly_demand is not None else scen["total_hourly_demand"]
    sumo_runner = SUMOEvaluatorImproved(
        sumo_config_path=scen["sumo_cfg"],
        total_hourly_demand=demand,
        scenario_id=scenario,
    )
    problem_kwargs = {}
    if cycle_bounds is not None:
        problem_kwargs["cycle_bounds"] = cycle_bounds
    if green_bounds is not None:
        problem_kwargs["green_bounds"] = green_bounds
    problem = ProblemV2_HonestObjective(evaluator=sumo_runner, **problem_kwargs)
    repair_v2 = TrafficSignalRepairImproved()

    algorithm = NSGA2WithDiversityInfill(
        pop_size=pop_size,
        max_retries=5,
        repair=repair_v2,
        sampling=RepairedSampling(repair_v2),
        crossover=SBX(prob=0.9, eta=15, repair=repair_v2, vtype=float),
        mutation=PM(prob=0.2, eta=20, repair=repair_v2, vtype=float),
        eliminate_duplicates=True,
    )

    early_stop = HVEarlyStopping(patience=5, min_delta=0.005)
    normalizer = RunningNormalizer(n_obj=2)
    algorithm.setup(problem, termination=("n_gen", n_gen_max), seed=seed)

    history_data: list = []
    while algorithm.has_next():
        if should_cancel is not None and should_cancel():
            print(f"[HUỶ] Job kịch bản {scenario} bị huỷ ở thế hệ {algorithm.n_gen}")
            break

        algorithm.next()
        pop = algorithm.pop
        F = pop.get("F")
        rank = pop.get("rank")
        feasible = pop.get("feasible")

        mask_valid = feasible.flatten() if feasible is not None else np.all(F < 90000, axis=1)
        F_valid = F[mask_valid]
        if len(F_valid) > 0:
            normalizer.update(F_valid)

        if rank is not None:
            rank_safe = np.array([r if r is not None else 999 for r in rank.flatten()], dtype=int)
            mask_front = (rank_safe == 0) & mask_valid
        else:
            mask_front = np.zeros(len(F), dtype=bool)
        F_front = F[mask_front]

        if len(F_front) > 0:
            hv_info = early_stop.update(F_front, normalizer)
        else:
            early_stop.no_improve += 1
            hv_info = {
                "hv": early_stop.best_hv if early_stop.best_hv != -np.inf else 0.0,
                "improved": False,
                "should_stop": early_stop.no_improve >= early_stop.patience,
                "no_improve": early_stop.no_improve,
            }

        record_and_print_gen(algorithm, history_data, hv_info, scenario)

        if on_generation is not None:
            on_generation(
                GenerationEvent(
                    scenario=scenario,
                    gen=algorithm.n_gen,
                    n_gen_max=n_gen_max,
                    hv=hv_info["hv"],
                    improved=hv_info["improved"],
                    pareto_size=int(mask_front.sum()),
                )
            )

        if algorithm.n_gen % 5 == 0:
            algorithm.log_diversity_stats()

        if hv_info["should_stop"]:
            reason = (
                f"Early Stopping (HV không cải thiện > "
                f"{early_stop.min_delta * 100:.1f}% trong {early_stop.patience} Gen)"
            )
            print(f"\n[DỪNG SỚM] {reason}")
            if on_generation is not None:
                on_generation(
                    GenerationEvent(
                        scenario=scenario,
                        gen=algorithm.n_gen,
                        n_gen_max=n_gen_max,
                        hv=hv_info["hv"],
                        improved=False,
                        pareto_size=int(mask_front.sum()),
                        stopped=True,
                        stop_reason=reason,
                    )
                )
            break

    algorithm.log_diversity_stats()

    columns = [
        "Gen", "C", "g1_1", "g1_2", "g2_1", "g2_2", "Offset",
        "f1_mean_timeloss_s_per_veh", "f2_co2_kg_per_veh", "Rank",
        "CrowdingDistance", "Scenario",
    ]
    df = pd.DataFrame(history_data, columns=columns)
    hv_map = {g + 1: v for g, v in enumerate(early_stop.hv_history)}
    df["HV_gen"] = df["Gen"].map(hv_map).fillna(0.0)

    # EDGE CASE (phát hiện qua tests/test_jobs.py::test_cancel_pending_job):
    # nếu should_cancel() trả True NGAY LẦN KIỂM TRA ĐẦU TIÊN — tức job bị
    # huỷ trước khi algorithm.next() chạy dù chỉ một lần — pymoo chưa từng
    # gán algorithm.n_gen (vẫn là None). Không ép về 0/1 giả vờ "đã chạy",
    # mà giữ nguyên None và để lớp gọi (JobManager) tự quyết định: một job
    # không có generation nào thì không có gì để lưu CSV/PNG.
    gen_final = algorithm.n_gen  # có thể là None nếu bị huỷ trước generation đầu tiên

    return OptimizationResult(
        scenario=scenario,
        label=scen["label"],
        history_df=df,
        hv_history=early_stop.hv_history,
        gen_final=gen_final,
    )


def save_results(result: OptimizationResult, results_dir: str) -> OptimizationResult:
    """Ghi CSV + PNG ra đĩa từ một OptimizationResult đã có sẵn.

    Tách khỏi run_optimization() để lớp gọi (CLI hoặc service) tự quyết
    định VỊ TRÍ lưu — ví dụ service có thể lưu theo job_id thay vì theo
    scenario để tránh nhiều job cùng kịch bản ghi đè lên nhau.

    Args:
        result: Kết quả trả về từ run_optimization().
        results_dir: Thư mục đích để lưu CSV và PNG.

    Returns:
        Chính `result` đã được set csv_path/plot_path.

    Raises:
        ValueError: Nếu result.gen_final là None (job bị huỷ trước khi
            chạy generation nào — không có gì để vẽ/lưu). Lớp gọi nên
            kiểm tra `len(result.history_df) > 0` TRƯỚC khi gọi hàm này
            thay vì dựa vào exception này (xem core/jobs.py._run_job).
    """
    if result.gen_final is None or len(result.history_df) == 0:
        raise ValueError(
            "Không thể lưu kết quả: job không có generation nào hoàn thành "
            "(có thể đã bị huỷ ngay trước khi bắt đầu). Kiểm tra "
            "len(result.history_df) > 0 trước khi gọi save_results()."
        )
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = f"results_{result.scenario}_{result.label}_gen{result.gen_final}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    result.history_df.to_csv(csv_path, index=False)
    print(f"\n[THÀNH CÔNG] Đã lưu file: {csv_path}")

    plot_path = plot_results(
        df=result.history_df,
        scenario_id=result.scenario,
        label=result.label,
        out_dir=results_dir,
        hv_history=result.hv_history,
        gen_final=result.gen_final,
    )

    result.csv_path = csv_path
    result.plot_path = plot_path
    return result
