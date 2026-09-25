# -*- coding: utf-8 -*-
# run_nsga2.py
import os
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

# Import từ project của bạn
from config import SCENARIOS
from sumo_evaluator_ver2 import SUMOEvaluatorImproved
from repair_operator_ver2 import TrafficSignalRepairImproved
from problem_v2_honest import ProblemV2_HonestObjective

class RepairedSampling(Sampling):
    def __init__(self, repair_operator):
        super().__init__()
        self.repair = repair_operator

    def _do(self, problem, n_samples, **kwargs):
        """Sinh n_samples nghiệm khả thi unique ngay từ đầu.
        Thử tối đa n_samples × 10 lần; nếu không đủ unique → điền random (chấp nhận trùng).
        """
        seen = set()
        result = []
        max_attempts = n_samples * 10
        attempts = 0
        while len(result) < n_samples and attempts < max_attempts:
            x = np.random.randint(
                problem.xl.astype(int),
                problem.xu.astype(int) + 1,
                (1, problem.n_var)
            ).astype(float)
            x = self.repair._do(problem, x, **kwargs)
            key = tuple(int(round(v)) for v in x[0])
            if key not in seen:
                seen.add(key)
                result.append(x[0])
            attempts += 1
        # Fallback: nếu không đủ unique sau max_attempts → chấp nhận trùng
        while len(result) < n_samples:
            x = np.random.randint(
                problem.xl.astype(int),
                problem.xu.astype(int) + 1,
                (1, problem.n_var)
            ).astype(float)
            x = self.repair._do(problem, x, **kwargs)
            result.append(x[0])
        return np.array(result)

class HVEarlyStopping:
    """
    Early stopping dua tren Hypervolume voi he quy chieu dong.

    Van de cu: RunningNormalizer gian ra khi tim duoc ky luc moi.
    Khi f_min giam, toa do chuan hoa cua front cu bi "day" lai, HV tinh lai
    bi teo nho gia tao -> bo dem no_improve tang sai -> dung oan.

    Fix: Luu lai F_raw (toa do goc) cua front tot nhat. Moi lan so sanh,
    tai chuan hoa F_raw bang normalizer hien tai truoc khi tinh HV.
    -> Dam bao ca hai duoc do tren cung he quy chieu.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.001, ref_point: np.ndarray = None):
        self.patience   = patience
        self.min_delta  = min_delta
        self.ref_point  = ref_point if ref_point is not None else np.array([1.1, 1.1])
        self.hv_calc    = HV(ref_point=self.ref_point)
        self.best_hv    = -np.inf
        self.best_F_raw = None   # FIXED: luu toa do goc (truoc chuan hoa) cua front tot nhat
        self.no_improve = 0
        self.hv_history = []

    def update(self, F_raw: np.ndarray, normalizer) -> dict:
        """
        F_raw    : Pareto front hien tai -- toa do CHUA chuan hoa  [s/xe, kg/xe]
        normalizer: RunningNormalizer hien tai -- da duoc update voi F_raw nay
        """
        # Chuan hoa front hien tai
        F_norm_curr = normalizer.normalize(F_raw)
        F_norm_curr = F_norm_curr[np.all(F_norm_curr <= 1.05, axis=1)]

        if len(F_norm_curr) == 0:
            self.no_improve += 1
            return {
                "hv": max(0.0, self.best_hv if self.best_hv != -np.inf else 0.0),
                "improved": False,
                "should_stop": self.no_improve >= self.patience,
                "no_improve": self.no_improve
            }

        current_hv = self.hv_calc.do(F_norm_curr)
        self.hv_history.append(current_hv)

        if self.best_F_raw is None:
            # Gen dau: luon cap nhat
            improved = True
        else:
            # FIXED: tai chuan hoa front cu bang he quy chieu HIEN TAI
            F_norm_best = normalizer.normalize(self.best_F_raw)
            F_norm_best = F_norm_best[np.all(F_norm_best <= 1.05, axis=1)]
            adjusted_best_hv = self.hv_calc.do(F_norm_best) if len(F_norm_best) > 0 else 0.0
            improved = (current_hv - adjusted_best_hv) / (abs(adjusted_best_hv) + 1e-12) > self.min_delta

        if improved:
            self.best_hv    = current_hv
            self.best_F_raw = F_raw.copy()   # luu toa do goc
            self.no_improve = 0
        else:
            self.no_improve += 1

        return {
            "hv"         : current_hv,
            "improved"   : improved,
            "should_stop": self.no_improve >= self.patience,
            "no_improve" : self.no_improve
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
    
    # Lấy các mảng dữ liệu ra một lần
    X_arr = pop.get("X")
    F_arr = pop.get("F")
    rank_arr = pop.get("rank")
    crowding_arr = pop.get("crowding")
    
    for i, x in enumerate(X_arr):
        f1, f2 = F_arr[i]

        # 1. Rank — try/except vì rank_arr[i] có thể là numpy array shape (1,)
        try:
            r = int(rank_arr[i]) if rank_arr is not None else -1
        except (TypeError, ValueError, IndexError):
            r = -1

        # 2. Crowding Distance — tương tự, bảo vệ NoneType và array
        try:
            cr = float(crowding_arr[i]) if crowding_arr is not None else 0.0
        except (TypeError, ValueError, IndexError):
            cr = 0.0

        history_list.append([algorithm.n_gen, *map(int, x), float(f1), float(f2), r, cr, scenario_id])
        
    print(f"Hoàn thành Thế hệ {algorithm.n_gen:^2} | HV={hv_info['hv']:.6f} | Cải thiện: {'CÓ' if hv_info['improved'] else 'KHÔNG'}")
    
class NSGA2WithDiversityInfill(NSGA2):
    """
    NSGA-II với cơ chế retry sinh offspring để chống trùng lặp.

    Vấn đề gốc: không gian khả thi ℤ⁶ bị ép xuống 4 chiều tự do bởi
    2 ràng buộc đẳng thức (h₁=0, h₂=0) → SBX+PM+Repair hay chiếu về cùng
    điểm → trùng lặp nghiêm trọng trong Q_t.
    pymoo's eliminate_duplicates chỉ lọc giữa P_t và Q_t, không lọc trong Q_t.

    Giải pháp: override _infill() — sau khi sinh Q_t, phát hiện trùng (trong
    Q_t lẫn với P_t), thay thế bằng nghiệm mới qua mutation mạnh hơn.
    Nếu sau max_retries vẫn không đủ unique → random sampling để lấp đầy,
    đảm bảo thuật toán không bị treo.

    Parameters
    ----------
    max_retries : int
        Số lần thử tối đa để tìm nghiệm unique qua PM mạnh. Default = 5.
    repair : TrafficSignalRepairImproved | None
        Repair operator — dùng trong random fill để đảm bảo khả thi.
    """

    def __init__(self, *args, max_retries: int = 5, repair=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries    = max_retries
        self.repair         = repair   # lưu lại để dùng trong _infill
        self._dup_replaced  = 0
        self._random_filled = 0

    def _infill(self):
        """
        Sinh offspring Q_t, thay thế nghiệm trùng bằng nghiệm unique.

        Luồng:
        1. Gọi super()._infill() để lấy Q_t mặc định từ SBX+PM.
        2. Xây known_X = {X ∈ P_t} ∪ {X đã unique trong Q_t}.
        3. Với mỗi cá thể trong Q_t bị trùng:
           a. Thử mutation mạnh (eta giảm dần) trên parent ngẫu nhiên từ P_t.
           b. Áp Repair đảm bảo khả thi.
           c. Nếu unique → chấp nhận.
           d. Sau max_retries → random sampling + repair.
        4. Trả về Q_t đã làm sạch trùng lặp.
        """
        # ── Bước 1: Sinh offspring mặc định từ SBX + PM ──────────────────────
        off   = super()._infill()       # Population shape (n_off, n_var)
        pop_X = self.pop.get("X")       # P_t, shape (N, n_var)

        # ── Bước 2: Xây tập known (set of tuple, O(1) lookup) ────────────────
        def _to_key(x):
            return tuple(int(round(v)) for v in x)

        known         = set(_to_key(row) for row in pop_X)
        off_X         = off.get("X")
        n             = len(off_X)
        new_X         = []
        accepted_keys = set()   # tránh trùng trong chính Q_t

        for i in range(n):
            key = _to_key(off_X[i])
            if key not in known and key not in accepted_keys:
                # Nghiệm unique → giữ nguyên
                new_X.append(off_X[i].copy())
                accepted_keys.add(key)
            else:
                # ── Bước 3: Retry với mutation mạnh hơn ──────────────────
                found = False
                for attempt in range(self.max_retries):
                    parent_idx = np.random.randint(len(pop_X))
                    candidate  = pop_X[parent_idx].copy().reshape(1, -1)

                    # Eta giảm dần theo attempt để tăng biên độ đột biến
                    eta_m     = max(1, 20 - attempt * 3)
                    # pymoo lưu mutation trong self.mating.mutation, không phải self.mutation
                    _repair_op = getattr(self.mating.mutation, "repair", None)
                    pm_strong = PM(
                        prob=1.0,   # đột biến mọi gen
                        eta=eta_m,
                        repair=_repair_op,
                        vtype=float
                    )
                    mutated = pm_strong._do(self.problem, candidate, algorithm=self)
                    new_key = _to_key(mutated[0])

                    if new_key not in known and new_key not in accepted_keys:
                        new_X.append(mutated[0].copy())
                        accepted_keys.add(new_key)
                        self._dup_replaced += 1
                        found = True
                        break

                if not found:
                    # ── Bước 3d: Random fill ─────────────────────────────
                    for _ in range(20):
                        rand_x = np.random.randint(
                            self.problem.xl.astype(int),
                            self.problem.xu.astype(int) + 1
                        ).astype(float)
                        if self.repair is not None:
                            rand_x = self.repair._do(
                                self.problem, rand_x.reshape(1, -1)
                            )[0]
                        rk = _to_key(rand_x)
                        if rk not in known and rk not in accepted_keys:
                            new_X.append(rand_x)
                            accepted_keys.add(rk)
                            self._random_filled += 1
                            found = True
                            break

                    if not found:
                        # Worst case: chấp nhận trùng hơn là thiếu cá thể
                        new_X.append(off_X[i].copy())

        # ── Bước 4: Tái tạo Population với X đã làm sạch ─────────────────────
        new_off = Population.new("X", np.array(new_X))
        return new_off

    def log_diversity_stats(self):
        print(
            f"[DIVERSITY] Trùng lặp thay thế bằng mutation: {self._dup_replaced} | "
            f"Thay bằng random: {self._random_filled}"
        )


def plot_results(df: pd.DataFrame, scenario_id: str, label: str,
                 out_dir: str, hv_history: list, gen_final: int):
    """
    Vẽ 3 biểu đồ tổng kết sau khi NSGA-II kết thúc.

    Subplots:
        1. Không gian mục tiêu (f1 vs f2):
           - Điểm bị dominated (Rank > 0)  → xám nhạt
           - Pareto front (Rank = 0)        → đỏ, to hơn
           - Nghiệm lỗi/không hợp lệ (Rank = -1) → × vàng
        2. Lịch sử Hypervolume theo thế hệ
        3. Kích thước Pareto front (số nghiệm Rank=0) theo thế hệ

    Parameters
    ----------
    df          : DataFrame lịch sử, cột 'Rank', 'f1_mean_timeloss_s_per_veh',
                  'f2_co2_kg_per_veh', 'Gen'
    scenario_id : ví dụ "S2"
    label       : nhãn kịch bản, ví dụ "Duoi_bao_hoa_bat_doi_xung"
    out_dir     : thư mục lưu ảnh
    hv_history  : danh sách float HV per generation từ early_stop.hv_history
    gen_final   : thế hệ cuối cùng (algorithm.n_gen)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"NSGA-II | Kịch bản {scenario_id} ({label}) | Gen kết thúc: {gen_final}",
                 fontsize=13, fontweight="bold")

    # ── Subplot 1: Không gian mục tiêu ──────────────────────────────────────
    ax1 = axes[0]
    col_f1 = "f1_mean_timeloss_s_per_veh"  # giữ tên cột CSV (backward compat)
    col_f2 = "f2_co2_kg_per_veh"

    mask_error     = df["Rank"] == -1
    mask_dominated = df["Rank"] > 0
    mask_pareto    = df["Rank"] == 0

    if mask_dominated.any():
        ax1.scatter(df.loc[mask_dominated, col_f1],
                    df.loc[mask_dominated, col_f2],
                    c="lightgray", s=15, label="Dominated", zorder=1, alpha=0.6)
    if mask_pareto.any():
        ax1.scatter(df.loc[mask_pareto, col_f1],
                    df.loc[mask_pareto, col_f2],
                    c="red", s=40, label="Pareto (Rank=0)", zorder=3)
    if mask_error.any():
        ax1.scatter(df.loc[mask_error, col_f1],
                    df.loc[mask_error, col_f2],
                    c="gold", marker="x", s=50, label="Lỗi (Rank=-1)", zorder=2)

    ax1.set_xlabel("f1 — Control Delay (s/xe)", fontsize=10)
    ax1.set_ylabel("f2 — CO₂ (kg/xe)", fontsize=10)
    ax1.set_title("Không gian mục tiêu (f1 vs f2)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Subplot 2: Lịch sử Hypervolume ──────────────────────────────────────
    ax2 = axes[1]
    if hv_history:
        gens = list(range(1, len(hv_history) + 1))
        ax2.plot(gens, hv_history, marker="o", markersize=4,
                 linewidth=1.5, color="steelblue")
        ax2.set_xlabel("Thế hệ", fontsize=10)
        ax2.set_ylabel("Hypervolume (chuẩn hóa)", fontsize=10)
        ax2.set_title("Lịch sử Hypervolume", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "Không có dữ liệu HV", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11)
        ax2.set_title("Lịch sử Hypervolume", fontsize=11)

    # ── Subplot 3: Kích thước Pareto front theo thế hệ ──────────────────────
    ax3 = axes[2]
    pareto_size_per_gen = (
        df[df["Rank"] == 0]
        .groupby("Gen")
        .size()
        .reindex(range(1, gen_final + 1), fill_value=0)
    )
    if not pareto_size_per_gen.empty:
        ax3.bar(pareto_size_per_gen.index, pareto_size_per_gen.values,
                color="tomato", edgecolor="darkred", linewidth=0.5)
        ax3.set_xlabel("Thế hệ", fontsize=10)
        ax3.set_ylabel("Số nghiệm Rank=0", fontsize=10)
        ax3.set_title("Kích thước Pareto front theo thế hệ", fontsize=11)
        ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        ax3.text(0.5, 0.5, "Không có dữ liệu Rank=0", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=11)
        ax3.set_title("Kích thước Pareto front theo thế hệ", fontsize=11)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"results_{scenario_id}_{label}_gen{gen_final}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    # BẮT BUỘC NGƯỜI DÙNG PHẢI NHẬP SCENARIO (Bỏ default="S2" đi)
    parser.add_argument("--scenario", type=str, required=True, choices=["S1", "S2", "S3"], help="Bắt buộc nhập S1, S2 hoặc S3")
    parser.add_argument("--pop_size", type=int, default=60)
    parser.add_argument("--n_gen_max", type=int, default=30)
    args = parser.parse_args()

    # TỰ ĐỘNG NỘI SUY TỪ config.py
    scen = SCENARIOS[args.scenario]
    sumo_cfg_path = scen["sumo_cfg"]
    demand = scen["total_hourly_demand"]

    print(f"\n{'='*60}\nKỊCH BẢN: {args.scenario} ({scen['label']})\nLưu lượng: {demand} xe/h\nFile SUMO: {sumo_cfg_path}\n{'='*60}\n")

    sumo_runner = SUMOEvaluatorImproved(sumo_config_path=sumo_cfg_path, total_hourly_demand=demand, scenario_id=args.scenario)
    problem = ProblemV2_HonestObjective(evaluator=sumo_runner)
    repair_v2 = TrafficSignalRepairImproved()

    algorithm = NSGA2WithDiversityInfill(
        pop_size=args.pop_size,
        max_retries=5,
        repair=repair_v2,           # truyền để _infill dùng cho random fill
        sampling=RepairedSampling(repair_v2),
        crossover=SBX(prob=0.9, eta=15, repair=repair_v2, vtype=float),
        mutation=PM(prob=0.2, eta=20, repair=repair_v2, vtype=float),
        eliminate_duplicates=True,
    )

    early_stop = HVEarlyStopping(patience=5, min_delta=0.005)
    normalizer = RunningNormalizer(n_obj=2)
    algorithm.setup(problem, termination=("n_gen", args.n_gen_max), seed=42)

    history_data = []
    while algorithm.has_next():
        algorithm.next()
        pop = algorithm.pop
        
        # Lấy giá trị mục tiêu (F), rank và tính khả thi (feasible)
        F = pop.get("F")
        rank = pop.get("rank")
        feasible = pop.get("feasible") # Trả về mảng boolean (True/False)
        
        # 1. Chỉ cập nhật Min/Max bằng những nghiệm KHẢ THI (Không kẹt xe, không lỗi)
        # Nếu mảng feasible bị None (do pymoo version), fallback dùng F < 90000
        if feasible is not None:
            mask_valid = feasible.flatten()
        else:
            mask_valid = np.all(F < 90000, axis=1)

        F_valid = F[mask_valid]
        if len(F_valid) > 0:
            normalizer.update(F_valid)

        # 2. Lấy Pareto Front thực sự (Rank = 0 VÀ Khả thi) để tính Hypervolume
        if rank is not None:
            rank_safe = np.array([r if r is not None else 999 for r in rank.flatten()], dtype=int)
            mask_front = (rank_safe == 0) & mask_valid
        else:
            mask_front = np.zeros(len(F), dtype=bool)
        F_front = F[mask_front]

        if len(F_front) > 0:
            # FIXED: truyen F_front goc (chua chuan hoa) + normalizer vao update()
            # HVEarlyStopping tu chuan hoa noi bo, dam bao cung he quy chieu
            hv_info = early_stop.update(F_front, normalizer)
        else:
            early_stop.no_improve += 1
            hv_info = {
                "hv"         : early_stop.best_hv if early_stop.best_hv != -np.inf else 0.0,
                "improved"   : False,
                "should_stop": early_stop.no_improve >= early_stop.patience,
                "no_improve" : early_stop.no_improve
            }

        record_and_print_gen(algorithm, history_data, hv_info, args.scenario)

        # Log thống kê đa dạng mỗi 5 thế hệ
        if algorithm.n_gen % 5 == 0:
            algorithm.log_diversity_stats()

        if hv_info["should_stop"]:
            stop_reason = f"Early Stopping (HV không cải thiện > {early_stop.min_delta*100:.1f}% trong {early_stop.patience} Gen)"
            print(f"\n[DỪNG SỚM] {stop_reason}")
            break

    # Log tổng kết đa dạng sau khi vòng lặp kết thúc
    algorithm.log_diversity_stats()

    df = pd.DataFrame(history_data, columns=['Gen', 'C', 'g1_1', 'g1_2', 'g2_1', 'g2_2', 'Offset', 'f1_mean_timeloss_s_per_veh', 'f2_co2_kg_per_veh', 'Rank', 'CrowdingDistance', 'Scenario'])
    hv_map = {g+1: v for g, v in enumerate(early_stop.hv_history)}
    df["HV_gen"] = df["Gen"].map(hv_map).fillna(0.0)
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Tạo đường dẫn file tuyệt đối
    csv_filename = f"results_{args.scenario}_{scen['label']}_gen{algorithm.n_gen}.csv"
    out_csv = os.path.join(results_dir, csv_filename)
    
    df.to_csv(out_csv, index=False)
    print(f"\n[THÀNH CÔNG] Đã lưu file: {out_csv}")

    # Vẽ và lưu 3 biểu đồ tổng kết
    plot_results(
        df=df,
        scenario_id=args.scenario,
        label=scen["label"],
        out_dir=results_dir,
        hv_history=early_stop.hv_history,
        gen_final=algorithm.n_gen,
    )

if __name__ == "__main__":
    main()