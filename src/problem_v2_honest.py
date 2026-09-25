# -*- coding: utf-8 -*-  # FIXED L7: thêm coding declaration
"""
problem_v2_honest.py
────────────────────
Định nghĩa lớp bài toán MOIP cho pymoo, kết nối với SUMOEvaluatorImproved.

Bài toán:
    min  F(x) = (f1(x), f2(x))
    s.t. h1(x) = g1_1 + g1_2 + 8 - C = 0          (ràng buộc đẳng thức)
         h2(x) = g2_1 + g2_2 + 8 - C = 0          (ràng buộc đẳng thức)
         g(x)  = gridlock_cv(x)       ≤ 0         (ràng buộc bất đẳng thức)
         x ∈ ℤ⁶,  bounds xem xl/xu bên dưới

Biến quyết định (6 chiều):
    x = [C, g1_1, g1_2, g2_1, g2_2, O2]
    - C      ∈ [60, 120]   giây — chu kỳ đèn chung
    - g1_1   ∈ [15,  90]   giây — thời gian xanh pha 1 tại J1
    - g1_2   ∈ [15,  90]   giây — thời gian xanh pha 2 tại J1
    - g2_1   ∈ [15,  90]   giây — thời gian xanh pha 1 tại J2
    - g2_2   ∈ [15,  90]   giây — thời gian xanh pha 2 tại J2
    - O2     ∈ [0, C-1]    giây — độ lệch pha J2 (dynamic bound, xử lý bởi Repair Operator)

Hàm mục tiêu:
    f1 = d_control  [s/xe]  — Halting-based Queue Integral (Tích phân hàng chờ)
         Công thức:  d_control = Σ_{t=Twu}^{Tsim} halting(t)·Δt / N_demand / KAPPA
         - halting(t): số xe v < 0.1 m/s tại bước t — từ summary.xml
         - N_demand = total_hourly_demand × (Tsim-Twu)/3600 — HẰNG SỐ per kịch bản
         - KAPPA = 0.9 (TRB E-C014, 2001) — hệ số stopped → control delay, không đổi theo kịch bản
         - Không có survivorship bias: halting đếm tất cả xe dừng tại từng bước
         - Mẫu số N_demand là hằng số → xung đột Pareto thực sự với f2
         - Nguồn: evaluator.evaluate() index 0
    f2 = co2  [kg/xe]
         - Phát thải CO2 tích phân, chuẩn hóa theo cùng N_demand với f1.
         - Nguồn: emission.xml (index 1 trong tuple evaluator).

Ràng buộc bất đẳng thức (pymoo n_ieq_constr=1):
    G[0] = cv_gridlock — index 2 trong tuple evaluator.
    pymoo constrained dominance: nghiệm vi phạm G[0]>0 luôn thua nghiệm khả thi.

Lưu ý:
    evaluator.evaluate() trả về tuple 4 phần tử theo thứ tự:
        [0] f1_halting      [s/xe]  — halting queue integral / N_demand / KAPPA (DÙNG cho tối ưu)
        [1] f2_co2          [kg/xe] — CO2 / N_demand
        [2] cv_gridlock             — ràng buộc gridlock (>0 vi phạm)
        [3] f1_timeloss_val [s/xe]  — timeLoss / N_demand từ tripinfo, VALIDATION ONLY
"""

import numpy as np
from pymoo.core.problem import Problem


class ProblemV2_HonestObjective(Problem):
    """
    Bài toán MOIP tối ưu tín hiệu giao thông — phiên bản V2.

    Parameters
    ----------
    evaluator : SUMOEvaluatorImproved
        Đối tượng đánh giá SUMO. evaluate() trả về tuple 4 phần tử:
        (f1_control, f2_co2, cv_gridlock, delay_tripinfo_validation)
        Dùng: index 0 → f1, index 1 → f2, index 2 → constraint.
        Bỏ qua: index 3 (delay_tripinfo — validation only, có survivorship bias).
    """

    def __init__(self, evaluator):
        # n_var=6, n_obj=2, n_ieq_constr=1 (gridlock constraint)
        super().__init__(
            n_var=6,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.array([60.0, 15.0, 15.0, 15.0, 15.0,  0.0]),
            xu=np.array([120.0, 90.0, 90.0, 90.0, 90.0, 119.0]),
        )
        self.evaluator = evaluator

    # ------------------------------------------------------------------ #
    # Hàm đánh giá chính — pymoo gọi _evaluate() cho từng batch cá thể   #
    # ------------------------------------------------------------------ #
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_individuals, 6)
            Ma trận các cá thể cần đánh giá.
        out : dict
            pymoo điền vào:
                out["F"]  — ma trận mục tiêu  shape (n, 2)
                out["G"]  — ma trận ràng buộc shape (n, 1)
        """
        n = X.shape[0]
        F = np.full((n, 2), 99999.0)   # [f1, f2]
        G = np.full((n, 1), 99999.0)   # [cv_gridlock]

        for i in range(n):
            C, g1_1, g1_2, g2_1, g2_2, O2 = (int(round(v)) for v in X[i])

            # Gọi SUMO evaluator — trả về tuple:
            # (f1_control, f2_co2, cv_gridlock, delay_tripinfo_validation)
            results = self.evaluator.evaluate(C, g1_1, g1_2, g2_1, g2_2, O2)

            if len(results) >= 3:
                f1_halting  = results[0]   # halting queue integral [s/xe] — dùng cho tối ưu
                f2_co2      = results[1]   # phát thải CO2 [kg/xe]
                cv_gridlock = results[2]   # ràng buộc gridlock (>0 = vi phạm)
                # results[3] = f1_timeloss_val — validation only, KHÔNG dùng ở đây
            else:
                f1_halting = f2_co2 = cv_gridlock = 99999.0

            F[i, 0] = f1_halting   # f1: d_control = Σhalting·Δt/N_demand/KAPPA [s/xe]
            F[i, 1] = f2_co2       # f2: phát thải CO2 / N_demand [kg/xe]
            G[i, 0] = cv_gridlock  # g(x): dương = vi phạm, âm/0 = khả thi

        out["F"] = F
        out["G"] = G