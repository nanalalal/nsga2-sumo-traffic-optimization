"""
extract_results.py
==================
Trích xuất toàn bộ số liệu thực từ 3 file CSV kết quả NSGA-II.

Sửa bug Gen+1: CSV ghi Gen bắt đầu từ 2 (pymoo đánh số từ thế hệ 2).
Gen thực sự = Gen_CSV - 1  (thế hệ 1 = Gen_CSV 2, ...).

Output:
  1. Bảng HV_norm theo thế hệ  (thay Bảng 7 báo cáo)
  2. Bảng Pareto front cuối cùng + Trade-off Rate + Spacing SP
  3. Bảng tổng hợp chỉ số đánh giá (thay Bảng 10 báo cáo)
  4. File LaTeX sẵn dùng copy-paste
"""

import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV

# ─── Cấu hình ───────────────────────────────────────────────────────────────
FILES = {
    "S1": "C:/Users/Admin/results/results_S1_Duoi_bao_hoa_doi_xung_gen16.csv",
    "S2": "C:/Users/Admin/results/results_S2_Duoi_bao_hoa_bat_doi_xung_gen16.csv",
    "S3": "C:/Users/Admin/results/results_S3_Gan_bao_hoa_bat_doi_xung_gen30.csv",
}
LABELS = {
    "S1": "Dưới bão hòa, đối xứng",
    "S2": "Dưới bão hòa, bất đối xứng",
    "S3": "Gần bão hòa, bất đối xứng",
}

# Cận chuẩn hóa cố định (phù hợp báo cáo)
R_MIN = np.array([8.0, 0.20])
R_MAX = np.array([120.0, 0.45])
REF_POINT = np.array([1.1, 1.1])

F1_COL = "f1_mean_timeloss_s_per_veh"
F2_COL = "f2_co2_kg_per_veh"

hv_indicator = HV(ref_point=REF_POINT)


# ─── Hàm tiện ích ───────────────────────────────────────────────────────────

def normalize(f1, f2):
    """Chuẩn hóa với kẹp trần tại 1.0."""
    t1 = np.clip((f1 - R_MIN[0]) / (R_MAX[0] - R_MIN[0]), 0, 1)
    t2 = np.clip((f2 - R_MIN[1]) / (R_MAX[1] - R_MIN[1]), 0, 1)
    return t1, t2


def compute_hv(df_pareto):
    """Tính HV_norm từ tập Pareto (Rank==0, hợp lệ)."""
    if len(df_pareto) == 0:
        return 0.0
    t1, t2 = normalize(
        df_pareto[F1_COL].values,
        df_pareto[F2_COL].values,
    )
    F = np.column_stack([t1, t2])
    # Lọc điểm nằm trong hộp [0, ref_point]
    mask = np.all(F <= REF_POINT, axis=1)
    F = F[mask]
    if len(F) == 0:
        return 0.0
    return float(hv_indicator.do(F))


def compute_spacing(df_pareto):
    """
    Spacing metric SP (Schott 1995):
      SP = sqrt( 1/(n-1) * sum( (d_i - d_bar)^2 ) )
    với d_i = min khoảng cách L1 đến nghiệm gần nhất
    trong không gian mục tiêu chuẩn hóa.
    """
    if len(df_pareto) < 2:
        return None
    t1, t2 = normalize(
        df_pareto[F1_COL].values,
        df_pareto[F2_COL].values,
    )
    pts = np.column_stack([t1, t2])
    n = len(pts)
    d = []
    for i in range(n):
        dists = [
            np.sum(np.abs(pts[i] - pts[j]))
            for j in range(n) if j != i
        ]
        d.append(min(dists))
    d = np.array(d)
    d_bar = d.mean()
    sp = np.sqrt(np.sum((d - d_bar) ** 2) / (n - 1))
    return float(sp)


def tradeoff_rate(df_pareto):
    """
    Trade-off Rate tổng hợp:
      TR_bar = |f2_max - f2_min| / |f1_max - f1_min|
    Và TR cục bộ giữa các nghiệm liền kề (sắp xếp theo f1).
    """
    if len(df_pareto) < 2:
        return None, None
    srt = df_pareto.sort_values(F1_COL).reset_index(drop=True)
    delta_f1 = srt[F1_COL].max() - srt[F1_COL].min()
    delta_f2 = abs(srt[F2_COL].max() - srt[F2_COL].min())
    if delta_f1 < 1e-9:
        tr_global = 0.0
    else:
        tr_global = delta_f2 / delta_f1

    # Cục bộ
    local_tr = []
    for i in range(len(srt) - 1):
        df1 = abs(srt.loc[i+1, F1_COL] - srt.loc[i, F1_COL])
        df2 = abs(srt.loc[i+1, F2_COL] - srt.loc[i, F2_COL])
        if df1 > 1e-9:
            local_tr.append(df2 / df1)
    return tr_global, local_tr


# ─── Xử lý từng kịch bản ────────────────────────────────────────────────────

results = {}

for sid, fpath in FILES.items():
    df = pd.read_csv(fpath)

    # ── Sửa bug Gen+1: Gen thực = Gen_CSV - 1 ────────────────────────────
    df["Gen_real"] = df["Gen"] - 1

    # Lọc nghiệm hợp lệ (loại Rank == -1 là lỗi SUMO)
    df_valid = df[df["Rank"] >= 0].copy()

    gen_max_csv = df["Gen"].max()
    gen_max_real = gen_max_csv - 1

    # ── HV_norm theo từng thế hệ (thực) ──────────────────────────────────
    hv_by_gen = {}
    for gen_csv in sorted(df["Gen"].unique()):
        gen_real = gen_csv - 1
        sub = df_valid[(df_valid["Gen"] == gen_csv) & (df_valid["Rank"] == 0)]
        hv_by_gen[gen_real] = compute_hv(sub)

    # ── Pareto front cuối (thế hệ thực cuối) ─────────────────────────────
    df_final_pareto = df_valid[
        (df_valid["Gen"] == gen_max_csv) & (df_valid["Rank"] == 0)
    ].copy()

    # Loại bỏ trùng lặp phenotype
    df_final_pareto = df_final_pareto.drop_duplicates(
        subset=["C", "g1_1", "g1_2", "g2_1", "g2_2", "Offset"]
    ).sort_values(F1_COL).reset_index(drop=True)

    # Sửa Gen hiển thị
    df_final_pareto["Gen_real"] = df_final_pareto["Gen"] - 1

    n_pareto = len(df_final_pareto)
    sp_val   = compute_spacing(df_final_pareto)
    hv_final = hv_by_gen.get(gen_max_real, 0.0)
    tr_global, tr_local = tradeoff_rate(df_final_pareto)

    delta_f1 = df_final_pareto[F1_COL].max() - df_final_pareto[F1_COL].min()
    delta_f2 = abs(df_final_pareto[F2_COL].max() - df_final_pareto[F2_COL].min())

    results[sid] = {
        "df": df,
        "df_valid": df_valid,
        "df_pareto": df_final_pareto,
        "hv_by_gen": hv_by_gen,
        "gen_max_real": gen_max_real,
        "n_pareto": n_pareto,
        "hv_final": hv_final,
        "sp": sp_val,
        "tr_global": tr_global,
        "tr_local": tr_local,
        "delta_f1": delta_f1,
        "delta_f2": delta_f2,
    }

# ─── In kết quả ─────────────────────────────────────────────────────────────

print("=" * 70)
print("BẢNG 1: HV_norm theo thế hệ THỰC (đã sửa Gen+1)")
print("=" * 70)

# Thu thập tất cả gen thực
all_gens = set()
for r in results.values():
    all_gens.update(r["hv_by_gen"].keys())
all_gens = sorted(all_gens)

header = f"{'Gen_thực':>8}  {'HV-S1':>8}  {'HV-S2':>8}  {'HV-S3':>8}"
print(header)
print("-" * len(header))
for g in all_gens:
    s1 = results["S1"]["hv_by_gen"].get(g, float("nan"))
    s2 = results["S2"]["hv_by_gen"].get(g, float("nan"))
    s3 = results["S3"]["hv_by_gen"].get(g, float("nan"))
    s1s = f"{s1:.4f}" if not np.isnan(s1) else "  --  "
    s2s = f"{s2:.4f}" if not np.isnan(s2) else "  --  "
    s3s = f"{s3:.4f}" if not np.isnan(s3) else "  --  "
    print(f"{g:>8}  {s1s:>8}  {s2s:>8}  {s3s:>8}")

print()
print("=" * 70)
print("BẢNG 2: Tổng hợp chỉ số đánh giá cuối cùng")
print("=" * 70)
print(f"{'KB':>4}  {'|Y_N|':>6}  {'HV_norm':>8}  {'Δf1 (s/xe)':>12}  "
      f"{'Δf2 (kg/xe)':>12}  {'TR_bar (kg/s)':>14}  {'SP':>8}")
print("-" * 75)
for sid in ["S1","S2","S3"]:
    r = results[sid]
    sp_str = f"{r['sp']:.4f}" if r['sp'] is not None else " --"
    tr_str = f"{r['tr_global']:.6f}" if r['tr_global'] is not None else " --"
    print(f"{sid:>4}  {r['n_pareto']:>6}  {r['hv_final']:>8.4f}  "
          f"{r['delta_f1']:>12.4f}  {r['delta_f2']:>12.6f}  "
          f"{tr_str:>14}  {sp_str:>8}")

print()
print("=" * 70)
print("BẢNG 3: Tập Pareto front cuối cùng — từng nghiệm")
print("=" * 70)
for sid in ["S1","S2","S3"]:
    r = results[sid]
    print(f"\n--- {sid} ({LABELS[sid]}) | Gen thực = {r['gen_max_real']} | "
          f"|Y_N| = {r['n_pareto']} ---")
    cols = ["C","g1_1","g1_2","g2_1","g2_2","Offset",F1_COL,F2_COL]
    print(r["df_pareto"][cols].to_string(index=False))

print()
print("=" * 70)
print("BẢNG 4: Trade-off Rate cục bộ — S3")
print("=" * 70)
r3 = results["S3"]
srt3 = r3["df_pareto"].sort_values(F1_COL).reset_index(drop=True)
print(f"{'ID':>4}  {'f1 (s/xe)':>12}  {'f2 (kg/xe)':>12}  "
      f"{'TR_local (kg/s)':>16}")
print("-" * 50)
for i in range(len(srt3)):
    f1v = srt3.loc[i, F1_COL]
    f2v = srt3.loc[i, F2_COL]
    if i < len(srt3)-1:
        df1 = abs(srt3.loc[i+1, F1_COL] - f1v)
        df2 = abs(srt3.loc[i+1, F2_COL] - f2v)
        tr_l = f"{df2/df1:.6f}" if df1 > 1e-9 else "  --"
    else:
        tr_l = "  --"
    print(f"{i+1:>4}  {f1v:>12.4f}  {f2v:>12.6f}  {tr_l:>16}")

# ─── Lưu CSV ─────────────────────────────────────────────────────────────────
# HV history
hv_rows = []
for g in all_gens:
    row = {"Gen_thuc": g}
    for sid in ["S1","S2","S3"]:
        row[f"HV_{sid}"] = results[sid]["hv_by_gen"].get(g, np.nan)
    hv_rows.append(row)
df_hv = pd.DataFrame(hv_rows)
df_hv.to_csv("D:/DoAn_NSGA2_VISSIM/results/hv_history.csv", index=False)

# Tổng hợp
summary_rows = []
for sid in ["S1","S2","S3"]:
    r = results[sid]
    summary_rows.append({
        "Kịch bản": sid,
        "|Y_N|": r["n_pareto"],
        "HV_norm": round(r["hv_final"], 4),
        "delta_f1_s_per_veh": round(r["delta_f1"], 4),
        "delta_f2_kg_per_veh": round(r["delta_f2"], 6),
        "TR_bar_kg_per_s": round(r["tr_global"], 6) if r["tr_global"] else None,
        "SP": round(r["sp"], 4) if r["sp"] else None,
        "Gen_cuoi": r["gen_max_real"],
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("D:/DoAn_NSGA2_VISSIM/results/summary_metrics.csv", index=False)

# Pareto fronts
for sid in ["S1","S2","S3"]:
    results[sid]["df_pareto"].to_csv(
        f"D:/DoAn_NSGA2_VISSIM/results/pareto_final_{sid}.csv", index=False
    )

print("\n[OK] Đã lưu: hv_history.csv, summary_metrics.csv, pareto_final_S*.csv")