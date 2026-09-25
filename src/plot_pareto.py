"""
plot_pareto_v2.py
=================
Vẽ lại toàn bộ biểu đồ từ 3 file CSV kết quả NSGA-II.
Sửa bug Gen+1: Gen thực = Gen_CSV - 1.
Tính HV_norm đúng: ref point (1.1,1.1), chuẩn hóa Min-Max cố định.

Sinh 4 biểu đồ:
  1. Pareto front cuối (f1 vs f2) — 3 kịch bản side-by-side
  2. Lịch sử HV_norm theo thế hệ thực
  3. Kích thước |Y_N| theo thế hệ thực
  4. Trade-off Rate cục bộ — S3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pymoo.indicators.hv import HV

# ── Cấu hình ────────────────────────────────────────────────────────────────
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
COLORS = {"S1": "#2166ac", "S2": "#d6604d", "S3": "#1a9641"}

R_MIN  = np.array([8.0,  0.20])
R_MAX  = np.array([120.0, 0.45])
REF    = np.array([1.1, 1.1])
hv_ind = HV(ref_point=REF)

F1 = "f1_mean_timeloss_s_per_veh"
F2 = "f2_co2_kg_per_veh"


def norm(f1v, f2v):
    t1 = np.clip((f1v - R_MIN[0]) / (R_MAX[0] - R_MIN[0]), 0, 1)
    t2 = np.clip((f2v - R_MIN[1]) / (R_MAX[1] - R_MIN[1]), 0, 1)
    return np.column_stack([t1, t2])


def calc_hv(df_p):
    if len(df_p) == 0:
        return 0.0
    F = norm(df_p[F1].values, df_p[F2].values)
    F = F[np.all(F <= REF, axis=1)]
    return float(hv_ind.do(F)) if len(F) > 0 else 0.0


# ── Bước 1: Đọc và chuẩn bị dữ liệu ────────────────────────────────────────
data = {}
for sid, fpath in FILES.items():
    df = pd.read_csv(fpath)
    df["Gen_real"] = df["Gen"] - 1          # sửa bug Gen+1
    df_v = df[df["Rank"] >= 0].copy()       # loại lỗi SUMO

    gen_max = df["Gen"].max()
    gen_max_r = gen_max - 1

    # HV theo thế hệ
    hv_hist = {}
    for gc in sorted(df["Gen"].unique()):
        sub = df_v[(df_v["Gen"] == gc) & (df_v["Rank"] == 0)]
        hv_hist[gc - 1] = calc_hv(sub)     # key = gen thực

    # |Y_N| theo thế hệ
    pareto_size = (
        df_v[df_v["Rank"] == 0]
        .groupby("Gen")
        .size()
        .rename(lambda g: g - 1)            # gen thực
    )

    # Pareto front cuối
    df_p = df_v[(df_v["Gen"] == gen_max) & (df_v["Rank"] == 0)].copy()
    df_p = (df_p.drop_duplicates(subset=["C","g1_1","g1_2","g2_1","g2_2","Offset"])
               .sort_values(F1).reset_index(drop=True))

    data[sid] = {
        "df": df, "df_valid": df_v, "df_pareto": df_p,
        "hv_hist": hv_hist, "pareto_size": pareto_size,
        "gen_max_r": gen_max_r,
    }

# ── Biểu đồ 1: Pareto front 3 kịch bản ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Tập không bị trội $Y_N$ tại thế hệ dừng — Ba kịch bản",
             fontsize=13, fontweight="bold")

for ax, sid in zip(axes, ["S1","S2","S3"]):
    d = data[sid]
    gen_max_csv = d["df"]["Gen"].max()
    df_all = d["df_valid"]

    # Nền xám: toàn bộ cá thể bị dominated
    bg = df_all[df_all["Rank"] > 0]
    ax.scatter(bg[F1], bg[F2], c="lightgray", s=8, alpha=0.4, zorder=1,
               label="Dominated")

    # Pareto cuối — màu theo kịch bản
    dp = d["df_pareto"]
    ax.plot(dp[F1], dp[F2], color=COLORS[sid], linewidth=1.8, zorder=3)
    ax.scatter(dp[F1], dp[F2], color=COLORS[sid], s=55, zorder=4,
               edgecolors="black", linewidth=0.6,
               label=f"$Y_N$ ($t={d['gen_max_r']}$, $n={len(dp)}$)")

    # Đánh số nghiệm
    for i, row in dp.iterrows():
        ax.annotate(str(i+1), (row[F1], row[F2]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color="darkred")

    hv_v = d["hv_hist"].get(d["gen_max_r"], 0)
    ax.set_title(f"{sid} — {LABELS[sid]}\n"
                 f"$HV_{{\\mathrm{{norm}}}}={hv_v:.4f}$",
                 fontsize=10)
    ax.set_xlabel("$f_1$ — Độ trễ điều khiển (s/xe)", fontsize=9)
    ax.set_ylabel("$f_2$ — Phát thải CO$_2$ (kg/xe)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("D:/DoAn_NSGA2_VISSIM/results/fig1_pareto_fronts.png", dpi=180, bbox_inches="tight")
plt.close()
print("[OK] fig1_pareto_fronts.png")

# ── Biểu đồ 2: Lịch sử HV_norm ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Lịch sử $HV_{\\mathrm{norm}}$ theo thế hệ thực", fontsize=13,
             fontweight="bold")

markers = {"S1": "o", "S2": "s", "S3": "^"}
for sid in ["S1","S2","S3"]:
    hh = data[sid]["hv_hist"]
    gens = sorted(hh.keys())
    hvs  = [hh[g] for g in gens]
    ax.plot(gens, hvs, marker=markers[sid], linewidth=2.0,
            color=COLORS[sid], label=f"{sid} ({LABELS[sid]})",
            markersize=5)

ax.axhline(y=1.21, color="gray", linestyle=":", linewidth=1,
           label="$HV_{\\max} = 1.21$ (lý thuyết)")
ax.set_xlabel("Thế hệ thực $t$", fontsize=11)
ax.set_ylabel("$HV_{\\mathrm{norm}}$", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xticks(range(1, 30, 2))
plt.tight_layout()
plt.savefig("D:/DoAn_NSGA2_VISSIM/results/fig2_hv_history.png", dpi=180, bbox_inches="tight")
plt.close()
print("[OK] fig2_hv_history.png")

# ── Biểu đồ 3: Kích thước |Y_N| theo thế hệ ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
fig.suptitle("Kích thước $|Y_N|$ theo thế hệ thực", fontsize=13,
             fontweight="bold")

for ax, sid in zip(axes, ["S1","S2","S3"]):
    ps = data[sid]["pareto_size"]
    ax.bar(ps.index, ps.values, color=COLORS[sid],
           edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_title(f"{sid}", fontsize=11)
    ax.set_xlabel("Thế hệ thực $t$", fontsize=9)
    ax.set_ylabel("$|Y_N|$ (Rank=0)", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("D:/DoAn_NSGA2_VISSIM/results/fig3_pareto_size.png", dpi=180, bbox_inches="tight")
plt.close()
print("[OK] fig3_pareto_size.png")

# ── Biểu đồ 4: Trade-off Rate cục bộ — S3 ───────────────────────────────────
dp3 = data["S3"]["df_pareto"].reset_index(drop=True)
tr_local = []
pair_labels = []
for i in range(len(dp3)-1):
    df1 = abs(dp3.loc[i+1, F1] - dp3.loc[i, F1])
    df2 = abs(dp3.loc[i+1, F2] - dp3.loc[i, F2])
    tr_local.append(df2/df1 if df1 > 1e-9 else 0)
    pair_labels.append(f"{i+1}→{i+2}")

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8))
fig.suptitle("Phân tích Trade-off Rate cục bộ — Kịch bản S3",
             fontsize=13, fontweight="bold")

# Pareto front S3
dp3_srt = dp3.sort_values(F1).reset_index(drop=True)
ax_top.plot(dp3_srt[F1], dp3_srt[F2], "o-", color=COLORS["S3"],
            linewidth=2, markersize=7, markeredgecolor="black", markeredgewidth=0.5)
for i, row in dp3_srt.iterrows():
    ax_top.annotate(str(i+1), (row[F1], row[F2]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="darkgreen")
# Đánh dấu vùng knee (nghiệm 8-9)
ax_top.axvspan(dp3_srt.loc[7, F1]-0.5, dp3_srt.loc[8, F1]+0.5,
               alpha=0.15, color="orange", label="Knee region (8→9)")
ax_top.set_xlabel("$f_1$ (s/xe)", fontsize=10)
ax_top.set_ylabel("$f_2$ (kg/xe)", fontsize=10)
ax_top.set_title("Mặt trận Pareto S3 ($|Y_N|=13$)", fontsize=11)
ax_top.legend(fontsize=9)
ax_top.grid(True, linestyle="--", alpha=0.4)

# Trade-off Rate bars
bar_colors = ["#d62728" if v > 0.01 else COLORS["S3"] for v in tr_local]
bars = ax_bot.bar(range(len(tr_local)), tr_local, color=bar_colors,
                  edgecolor="black", linewidth=0.5)
# TR_global = |Δf2| / |Δf1| — độ dốc toàn cục, nhất quán với báo cáo
tr_global_val = (dp3_srt[F2].max() - dp3_srt[F2].min()) / \
                (dp3_srt[F1].max() - dp3_srt[F1].min())
ax_bot.axhline(y=tr_global_val, color="navy", linestyle="--",
               linewidth=1.5, label=f"$\\overline{{TR}}_{{\\mathrm{{global}}}}$ = {tr_global_val:.5f} kg/s")
ax_bot.set_xticks(range(len(pair_labels)))
ax_bot.set_xticklabels(pair_labels, rotation=45, fontsize=9)
ax_bot.set_xlabel("Cặp nghiệm liền kề", fontsize=10)
ax_bot.set_ylabel("$\\mathrm{TR}(i)$ (kg/s)", fontsize=10)
ax_bot.set_title("Trade-off Rate cục bộ giữa các cặp nghiệm", fontsize=11)
ax_bot.legend(fontsize=9)
ax_bot.grid(True, axis="y", linestyle="--", alpha=0.4)

# Annotation cho outlier
max_idx = int(np.argmax(tr_local))
ax_bot.annotate(f"Knee\n{tr_local[max_idx]:.3f}",
                xy=(max_idx, tr_local[max_idx]),
                xytext=(max_idx+0.5, tr_local[max_idx]*0.85),
                fontsize=8, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"))

plt.tight_layout()
plt.savefig("D:/DoAn_NSGA2_VISSIM/results/fig4_tradeoff_s3.png", dpi=180, bbox_inches="tight")
plt.close()
print("[OK] fig4_tradeoff_s3.png")

print("\n[DONE] Tất cả 4 biểu đồ đã được tạo.")
print("\nTóm tắt số liệu thực:")
print(f"{'KB':>4}  {'t*':>4}  {'|Y_N|':>6}  {'HV_norm':>8}  {'Δf1':>8}  {'Δf2':>10}")
for sid in ["S1","S2","S3"]:
    d = data[sid]
    dp = d["df_pareto"]
    hv_f = d["hv_hist"].get(d["gen_max_r"], 0)
    df1 = dp[F1].max() - dp[F1].min()
    df2 = abs(dp[F2].max() - dp[F2].min())
    print(f"{sid:>4}  {d['gen_max_r']:>4}  {len(dp):>6}  {hv_f:>8.4f}  "
          f"{df1:>8.3f}  {df2:>10.6f}")