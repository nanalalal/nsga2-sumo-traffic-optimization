# NSGA-II cho Điều khiển Tín hiệu Giao thông Phối hợp (MOIP) tích hợp SUMO

Đồ án tốt nghiệp: ứng dụng thuật toán tối ưu đa mục tiêu NSGA-II giải bài toán điều khiển tín hiệu giao thông phối hợp (Multi-Objective Intersection Planning), đánh giá bằng mô phỏng vi mô SUMO.

> **Repo liên quan:** lớp service triển khai (FastAPI + MCP Server + Docker) bọc quanh đồ án này nằm ở repo riêng — xem [nsga2-fastapi-mcp-service](https://github.com/nanalalal/nsga2-fastapi-mcp-service).

---

## Mục lục

1. [Giới thiệu tóm tắt](#1-giới-thiệu-tóm-tắt)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Hướng dẫn chạy](#4-hướng-dẫn-chạy)
5. [Các điểm nhấn kỹ thuật](#5-các-điểm-nhấn-kỹ-thuật)

---

## 1. Giới thiệu tóm tắt

Đồ án giải quyết **Bài toán Tối ưu Điều khiển Tín hiệu Giao thông Phối hợp** (Multi-Objective Intersection Planning — MOIP) cho nút giao hai pha có hai giao lộ liên quan (J1 và J2) sử dụng thuật toán tiến hóa đa mục tiêu **NSGA-II** (Non-dominated Sorting Genetic Algorithm II).

**Bài toán gốc** được phát biểu trên không gian quyết định vật lý 6 chiều (Phenotype):

$$\min_{\mathbf{x} \in \Omega \subset \mathbb{Z}^6} \; \mathbf{F}(\mathbf{x}) = \bigl(f_1(\mathbf{x}),\; f_2(\mathbf{x})\bigr)$$

$$\text{s.t.} \quad h_1(\mathbf{x}) = g_{1,1} + g_{1,2} + 8 - C = 0, \quad h_2(\mathbf{x}) = g_{2,1} + g_{2,2} + 8 - C = 0$$

$$g(\mathbf{x}) = r_{\text{halt}}(\mathbf{x}) - \theta_{\max} \leq 0, \quad \theta_{\max} = 0.6$$

Hai ràng buộc đẳng thức $h_1, h_2$ loại bỏ 2 chiều tự do, thu hẹp không gian khả thi xuống đa tạp affine $D_{\text{eff}} = 4$ chiều. Do đó, thuật toán NSGA-II thực tế **tìm kiếm trên không gian Genotype 4 chiều** $\mathbf{u} = [C,\, g_{1,1},\, g_{2,1},\, \theta_2]^T \in \mathbb{R}^4$ (kỹ thuật *Continuous Relaxation with Integer Projection*) và giải mã về Phenotype qua ánh xạ $\mathcal{M}: \mathbb{R}^4 \to \Omega \subset \mathbb{Z}^6$ trước khi gọi SUMO.

Trong đó:
- **$f_1(\mathbf{x})$** — Độ trễ điều khiển trung bình (s/xe), tính bằng tích phân hàng chờ (Queue Integral) từ dữ liệu `summary.xml` của SUMO. Không có survivorship bias.
- **$f_2(\mathbf{x})$** — Phát thải CO₂ trung bình (kg/xe), tích phân toàn thời gian mô phỏng từ `emission.xml`.
- **$\mathbf{x} = [C,\, g_{1,1},\, g_{1,2},\, g_{2,1},\, g_{2,2},\, O_2]$** — Vectơ biến quyết định 6 chiều.

Hàm mục tiêu được đánh giá hoàn toàn bằng **mô phỏng hộp đen** (black-box simulation) qua SUMO kết hợp giao diện điều khiển trực tuyến TraCI. Ba kịch bản lưu lượng đại diện được kiểm nghiệm:

| Ký hiệu | Lưu lượng (xe/h) | Đặc trưng | Thế hệ dừng | $\|Y_N\|$ |
|:---:|:---:|:---|:---:|:---:|
| **S1** | 2 000 | Dưới bão hòa, phân bổ đối xứng | 15 | 4 |
| **S2** | 2 000 | Dưới bão hòa, phân bổ bất đối xứng | 15 | 5 |
| **S3** | 3 600 | Gần bão hòa, phân bổ bất đối xứng | 29 | 13 |

> **Lưu ý về Gen CSV:** File CSV lưu `Gen` bắt đầu từ 2 (pymoo đánh số từ thế hệ thứ 2). **Thế hệ thực = Gen\_CSV − 1.**

---

## 2. Cấu trúc thư mục

```
DoAn_NSGA2_SUMO/
│
├── src/                          # Toàn bộ mã nguồn Python
│   ├── config.py                 # Tham số kịch bản
│   ├── problem_v2_honest.py      # Định nghĩa bài toán MOIP (pymoo)
│   ├── sumo_evaluator_ver2.py    # Hàm mục tiêu hộp đen (SUMO oracle)
│   ├── repair_operator_ver2.py   # Ánh xạ giải mã M (Repair + decode)
│   ├── run_nsga2_v2.py           # Bộ điều phối NSGA-II chính
│   ├── plot_pareto.py            # Trực quan hóa kết quả (4 biểu đồ)
│   └── extract_results.py        # Trích xuất & tính chỉ số từ CSV
│
├── sumo_model_s1/                # Mạng lưới SUMO — Kịch bản S1
│   ├── network.net.xml
│   ├── routes.rou.xml
│   ├── simulation.sumocfg
│   └── output/
│
├── sumo_model_s2/                # Mạng lưới SUMO — Kịch bản S2
├── sumo_model_s3/                # Mạng lưới SUMO — Kịch bản S3
│
└── results/                      # Kết quả CSV và hình ảnh đầu ra
    ├── results_S*_*_gen*.csv     # Lịch sử quần thể toàn bộ thế hệ
    ├── pareto_final_S*.csv       # Tập Pareto cuối mỗi kịch bản
    ├── hv_history.csv            # HV_norm theo thế hệ thực
    ├── summary_metrics.csv       # Chỉ số tổng hợp (HV, SP, TR_bar)
    └── fig*.png                  # 4 biểu đồ phân tích
```


### Vai trò toán học của từng file nguồn

#### `src/config.py` — Bảng tham số kịch bản

Định nghĩa từ điển `SCENARIOS` chứa các tham số đầu vào cố định cho mỗi kịch bản: tổng lưu lượng lý thuyết (xe/h) và đường dẫn tới file cấu hình SUMO. Tham số `total_hourly_demand` được dùng làm **mẫu số hằng** $N_{\text{demand}}$ trong công thức tính $f_1$ và $f_2$, đảm bảo tính nhất quán thứ nguyên giữa hai hàm mục tiêu.

---

#### `src/problem_v2_honest.py` — Định nghĩa bài toán MOIP

Lớp `ProblemV2_HonestObjective` kế thừa `pymoo.core.problem.Problem`, đóng vai trò **khung toán học của bài toán tối ưu**:

- Khai báo không gian quyết định $\mathbb{Z}^6$ với các cận:
  $C \in [60, 120]$, $g_{i,j} \in [15, 90]$, $O_2 \in [0, 119]$.
- Khai báo 2 hàm mục tiêu (`n_obj=2`) và 1 ràng buộc bất đẳng thức (`n_ieq_constr=1`).
- Phương thức `_evaluate()` gọi `SUMOEvaluatorImproved.evaluate()` và điền vào ma trận `out["F"]` và `out["G"]` mà pymoo yêu cầu.

> **Lưu ý:** Mặc dù lớp này khai báo không gian $\mathbb{Z}^6$, NSGA-II thực tế tìm kiếm trên $\mathbb{R}^4$ (Genotype) và chuyển về $\mathbb{Z}^6$ (Phenotype) qua `repair_operator_ver2.py` trước khi gọi `_evaluate()`.

---

#### `src/sumo_evaluator_ver2.py` — Hàm mục tiêu hộp đen (Black-box Oracle)

Lớp `SUMOEvaluatorImproved` là **cầu nối giữa không gian quyết định và không gian mục tiêu**, thực hiện toàn bộ việc đánh giá một cá thể $\mathbf{x}$ thông qua mô phỏng vi mô.

**Hàm mục tiêu $f_1$ — Tích phân hàng chờ (Queue Integral):**

$$f_1(\mathbf{x}) = \frac{1}{\kappa} \cdot \frac{\displaystyle\sum_{t=T_{wu}}^{T_{sim}} \text{halting}(t) \cdot \Delta t}{N_{\text{demand}}} \quad [\text{s/xe}]$$

Trong đó $\kappa = 0.9$ (TRB E-C014, 2001) là hệ số quy đổi stopped delay → control delay. Phương pháp này **không có survivorship bias** vì `halting(t)` đếm tất cả xe dừng tại từng bước thời gian, và mẫu số $N_{\text{demand}}$ là hằng số không phụ thuộc nghiệm $\mathbf{x}$.

**Hàm mục tiêu $f_2$ — Phát thải CO₂ trung bình:**

$$f_2(\mathbf{x}) = \frac{1}{10^6} \cdot \frac{\displaystyle\sum_{t=T_{wu}}^{T_{sim}} \sum_{v \in V(t)} E_{\text{CO}_2}(v,t) \cdot \Delta t}{N_{\text{demand}}} \quad [\text{kg/xe}]$$

**Ràng buộc gridlock $g(\mathbf{x})$:**

$$g(\mathbf{x}) = r_{\text{halt}}(\mathbf{x}) - \theta_{\max}, \quad r_{\text{halt}} = \frac{\sum_t \text{halting}(t)}{\sum_t \text{running}(t) + 1}$$

Số hạng $+1$ ở mẫu số bảo vệ khỏi chia cho 0 khi toàn bộ xe đứng yên (gridlock 100%).

**Xử lý lỗi thực thi SUMO:** Có hai nguồn phát sinh cờ $CV > 0$:
1. *Vi phạm ràng buộc vật lý* $g(\mathbf{x}) > 0$: SUMO hoàn thành, nghiệm vi phạm ngưỡng gridlock.
2. *Lỗi thực thi* (SUMO crash): $\mathbf{F}(\mathbf{x})$ không xác định (undefined). Cá thể nhận $CV = CV_{\max}$ để Constrained Domination đẩy xuống front cuối mà không can thiệp vào không gian mục tiêu.

File còn cài đặt cơ chế **retry** (tối đa `max_retries=3` lần với độ trễ 500 ms) và **circuit breaker** để phòng ngừa sự cố SUMO.

---

#### `src/repair_operator_ver2.py` — Ánh xạ giải mã $\mathcal{M}$ (Repair Operator)

Lớp `TrafficSignalRepairImproved` kế thừa `pymoo.core.repair.Repair`, thực hiện ánh xạ giải mã tất định từ không gian Genotype liên tục về Phenotype nguyên khả thi:

$$\mathcal{M}: \mathbf{u} \in \mathbb{R}^4 \longrightarrow \mathbf{x} \in \Omega \subset \mathbb{Z}^6$$

**Ba bước tuần tự:**

1. **Chiếu về số nguyên:** $\tilde{C} = \text{clip}(\lfloor C \rceil, 60, 120)$, $\tilde{g}_{i,1} = \text{clip}(\lfloor g_{i,1} \rceil, 15, 90)$.

2. **Hiệu chỉnh bảo toàn biên (Bounds-Preserving Correction):**  
   Tính phần dư $\Delta_i = \tilde{C} - (\tilde{g}_{i,1} + \tilde{g}_{i,2,\text{raw}} + 8)$ và phân bổ cho hai pha theo thứ tự ưu tiên ngẫu nhiên (50/50 giữa pha 1 và pha 2 trước), clip về $[15, 90]$.  
   Lý do dùng ngẫu nhiên: tránh mất đối xứng cấu trúc khi luôn ưu tiên cùng một pha.

3. **Ánh xạ độ lệch pha:** $O_2 \gets O_2 \bmod \tilde{C}$ để đảm bảo $O_2 \in [0, \tilde{C}-1]$.

**Fallback an toàn:** Nếu sau hiệu chỉnh nghiệm vẫn không khả thi (xác suất rất thấp), áp dụng cấu hình dự phòng $\mathbf{x}_{\text{fallback}} = [64, 20, 36, 20, 36, O_2']$ để tránh SUMO crash.

Toán tử được tích hợp vào ba điểm can thiệp: khởi tạo quần thể (`RepairedSampling`), lai ghép SBX và đột biến PM.

---

#### `src/run_nsga2_v2.py` — Bộ điều phối NSGA-II chính

File điều phối toàn bộ vòng lặp NSGA-II, bao gồm ba thành phần mở rộng:

| Lớp | Chức năng |
|:---|:---|
| `RepairedSampling` | Khởi tạo quần thể ban đầu đảm bảo tính khả thi và đa dạng (chống trùng lặp bằng tập `seen`, thử tối đa $10N$ lần) |
| `NSGA2WithDiversityInfill` | Override `_infill()`: thay thế offspring trùng bằng đột biến PM với $\eta$ giảm dần (biên độ tăng dần), fallback sang random sampling nếu vẫn trùng sau `max_retries=5` lần |
| `HVEarlyStopping` | Dừng sớm dựa trên $HV_{\text{norm}}$ với hệ quy chiếu động; lưu tọa độ gốc của front tốt nhất (`F_raw`) để tránh so sánh sai khi normalizer mở rộng — xem mục 5.3 |

**Điều kiện dừng sớm:** Dừng khi $HV_{\text{norm}}$ không cải thiện quá `min_delta=0.5%` trong `patience=5` thế hệ liên tiếp (sau giai đoạn warm-up `min_warmup_gen` thế hệ đầu).

---

#### `src/plot_pareto.py` — Trực quan hóa kết quả

Sinh 4 biểu đồ phân tích từ 3 file CSV kết quả (**sửa bug Gen+1** và dùng ref point $(1.1, 1.1)^T$ đúng):

1. **Tập không bị trội $Y_N$** — Pareto front cuối của cả ba kịch bản, có đánh số nghiệm.
2. **Lịch sử $HV_{\text{norm}}$** — Tiến trình hội tụ theo thế hệ thực, ref point $(1.1, 1.1)^T$.
3. **Kích thước $|Y_N|$** — Biến động số lượng nghiệm Pareto (Rank=0) theo thế hệ thực.
4. **Trade-off Rate cục bộ S3** — Phân tích $TR(i) = |\Delta f_2| / |\Delta f_1|$ giữa các nghiệm liền kề, xác định **vùng knee** tại cặp $8 \to 9$ ($TR = 0.161$ kg/s).

---

#### `src/extract_results.py` — Trích xuất chỉ số từ CSV

Script tính toán và xuất ra file tất cả chỉ số đánh giá:

- `hv_history.csv`: $HV_{\text{norm}}$ theo từng thế hệ thực cho cả 3 kịch bản.
- `summary_metrics.csv`: $|Y_N|$, $HV_{\text{norm}}$, $\Delta f_1$, $\Delta f_2$, $\overline{TR}$, $SP$ tại thế hệ dừng.
- `pareto_final_S*.csv`: Tập nghiệm Pareto cuối đầy đủ (đã loại trùng lặp Phenotype).

**Công thức các chỉ số được tính:**
- $HV_{\text{norm}}$: ref point $(1.1, 1.1)^T$, chuẩn hóa Min-Max cố định $[8.0, 120.0] \times [0.20, 0.45]$.
- $\overline{TR} = |\Delta f_2| / |\Delta f_1|$: tỷ lệ đánh đổi toàn cục.
- $SP = \sqrt{\frac{1}{|Y_N|-1} \sum_i (d_i - \bar{d})^2}$: Spacing metric (Schott 1995), $d_i$ = khoảng cách $L_1$ đến nghiệm gần nhất trong không gian chuẩn hóa.

---

#### `src/resume_nsga.py` — Resume từ checkpoint CSV

Script tiếp tục tối ưu từ một file CSV kết quả cũ với ba cải tiến chính so với `run_nsga2_v2.py`:

1. **HV tuyệt đối thay vì chuẩn hóa động** — Tính HV trực tiếp trên không gian gốc `[s/xe, kg/xe]` với ref point cố định `[120.0, 0.45]`, tránh hoàn toàn lỗi Reference Point Collapse.

2. **Warm-start từ Pareto front cũ** — `WarmStartSampling` khởi tạo quần thể bằng tập nghiệm Rank=0 của CSV cũ, bổ sung random fill cho phần còn lại.

3. **Checkpoint tự động mỗi 5 thế hệ + xử lý Ctrl+C an toàn** — `signal.SIGINT` handler lưu dữ liệu rồi thoát gracefully; nhấn Ctrl+C lần 2 mới force-exit.

```bash
python resume_nsga.py \
    --scenario S3 \
    --csv_old ../results/results_S3_Gan_bao_hoa_bat_doi_xung_gen30.csv \
    --n_gen_extra 40 \
    --patience 10
```

---

#### `sumo_model_s*/` — Mô hình mạng lưới SUMO

| File | Nội dung |
|:---|:---|
| `network.net.xml` | Cấu trúc hình học mạng lưới (đường, nút giao, làn xe) |
| `routes.rou.xml` | Kế hoạch phát sinh phương tiện (vehicle flows) theo kịch bản |
| `simulation.sumocfg` | File cấu hình tổng hợp của SUMO |

---

## 3. Yêu cầu hệ thống

### 3.1. Bộ mô phỏng SUMO

Tải và cài đặt **SUMO ≥ 1.15.0** tại: https://sumo.dlr.de/docs/Downloads.php

Sau khi cài, thiết lập biến môi trường `SUMO_HOME`:

**Windows — Command Prompt / Anaconda Prompt:**
```cmd
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
```

**Windows — PowerShell:**
```powershell
[System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
```

**Linux / macOS:**
```bash
export SUMO_HOME="/usr/share/sumo"
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
```

Kiểm tra:
```cmd
sumo --version
```

> `SUMOEvaluatorImproved` sẽ ném `EnvironmentError` ngay khi khởi tạo nếu `SUMO_HOME` chưa được khai báo.

---

### 3.2. Cài đặt Python và thư viện

Chọn **một trong hai cách** tùy môi trường làm việc của bạn:

---

#### Cách 1 — Anaconda (khuyến nghị cho Windows)

Dùng Anaconda nếu bạn đã có sẵn hoặc muốn tách biệt môi trường dễ dàng.

**Bước 1:** Tạo và kích hoạt môi trường (chỉ làm một lần):

```cmd
conda create -n traffic_nsga2 python=3.10
conda activate traffic_nsga2
```

**Bước 2:** Cài thư viện:

```cmd
pip install pymoo pandas matplotlib numpy scipy
pip install traci sumolib
```

Nếu `traci`/`sumolib` đã đi kèm SUMO nhưng Python chưa tìm thấy, thêm vào PATH:

```cmd
set PYTHONPATH=%SUMO_HOME%\tools;%PYTHONPATH%
```

**Bước 3:** Mỗi lần chạy dự án, mở **Anaconda Prompt** và kích hoạt:

```cmd
conda activate traffic_nsga2
```

---

#### Cách 2 — Python thuần + venv (không dùng Anaconda)

Dùng cách này nếu bạn chỉ có Python cài từ [python.org](https://www.python.org/downloads/) (≥ 3.9).

**Bước 1:** Tạo virtual environment:

```cmd
cd <đường-dẫn-bạn-clone-repo-vào>
python -m venv venv
```

**Bước 2:** Kích hoạt môi trường:

```cmd
# Windows — Command Prompt
venv\Scripts\activate.bat

# Windows — PowerShell
venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

**Bước 3:** Cài thư viện:

```cmd
pip install pymoo pandas matplotlib numpy scipy
pip install traci sumolib
```

**Bước 4:** Mỗi lần chạy dự án, nhớ kích hoạt lại venv trước (bước 2).

---

#### Thư viện cần thiết

| Thư viện | Phiên bản khuyến nghị | Mục đích |
|:---|:---:|:---|
| `pymoo` | ≥ 0.6.0 | Framework NSGA-II, toán tử di truyền, chỉ số HV |
| `traci` | cùng phiên bản SUMO | Điều khiển mô phỏng SUMO trực tuyến qua TraCI |
| `sumolib` | cùng phiên bản SUMO | Đọc binary SUMO |
| `pandas` | ≥ 1.5.0 | Xử lý dữ liệu CSV kết quả |
| `matplotlib` | ≥ 3.6.0 | Trực quan hóa Pareto front và lịch sử HV |
| `numpy` | ≥ 1.23.0 | Phép tính số học trên mảng |
| `scipy` | ≥ 1.9.0 | Phụ thuộc của pymoo |

---

## 4. Hướng dẫn chạy

> ⚠️ Luôn đảm bảo đã **kích hoạt môi trường** (conda hoặc venv) trước khi chạy bất kỳ lệnh nào. Chạy Python hệ thống sẽ gặp lỗi `ModuleNotFoundError: No module named 'pymoo'`.

### 4.1. Chạy NSGA-II từ đầu

**Cách 1 — Anaconda:**
```cmd
conda activate traffic_nsga2
cd <đường-dẫn-bạn-clone-repo-vào>\src
python run_nsga2_v2.py --scenario S1 --pop_size 60 --n_gen_max 30
```

**Cách 2 — venv:**
```cmd
cd <đường-dẫn-bạn-clone-repo-vào>
venv\Scripts\activate.bat
cd src
python run_nsga2_v2.py --scenario S1 --pop_size 60 --n_gen_max 30
```

Hoặc dùng **đường dẫn đầy đủ** (sau khi đã kích hoạt môi trường):
```cmd
python "<đường-dẫn-bạn-clone-repo-vào>\src\run_nsga2_v2.py" --scenario S1 --pop_size 60 --n_gen_max 30
```

Ba kịch bản:
```cmd
python run_nsga2_v2.py --scenario S1 --pop_size 60 --n_gen_max 30
python run_nsga2_v2.py --scenario S2 --pop_size 60 --n_gen_max 30
python run_nsga2_v2.py --scenario S3 --pop_size 60 --n_gen_max 50
```

| Tham số | Kiểu | Mặc định | Mô tả |
|:---|:---:|:---:|:---|
| `--scenario` | `str` | *(bắt buộc)* | Kịch bản: `S1`, `S2` hoặc `S3` |
| `--pop_size` | `int` | `60` | Kích thước quần thể $N$ |
| `--n_gen_max` | `int` | `30` | Số thế hệ tối đa (có thể dừng sớm hơn do early stopping) |

Kết quả lưu vào `results/results_{scenario}_{label}_gen{t}.csv` + biểu đồ PNG.

> **Thời gian ước tính:** $60 \times 30 = 1800$ lần gọi SUMO, mỗi lần ≈ 4 giây → **≈ 2 giờ CPU** mỗi kịch bản.

### 4.2. Resume từ checkpoint

```cmd
cd <đường-dẫn-bạn-clone-repo-vào>\src
python resume_nsga.py --scenario S3 --csv_old ..\results\results_S3_Gan_bao_hoa_bat_doi_xung_gen30.csv --n_gen_extra 40 --patience 10
```

Nhấn `Ctrl+C` một lần để lưu dữ liệu an toàn rồi thoát.

### 4.3. Trích xuất chỉ số và vẽ biểu đồ

Không cần SUMO — chỉ cần 3 file CSV trong `results/`:

```cmd
cd <đường-dẫn-bạn-clone-repo-vào>\src
python extract_results.py
python plot_pareto.py
```

> Kiểm tra đường dẫn `FILES` trong hai script này trỏ đúng đến file CSV trên máy của bạn trước khi chạy.

### 4.4. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|:---|:---|:---|
| `ModuleNotFoundError: No module named 'pymoo'` | Chưa kích hoạt môi trường | Chạy `conda activate traffic_nsga2` hoặc `venv\Scripts\activate.bat` |
| `No such file or directory: 'run_nsga2_v2.py'` | Sai thư mục làm việc | `cd <đường-dẫn-bạn-clone-repo-vào>\src` trước khi chạy |
| `EnvironmentError: Vui lòng khai báo SUMO_HOME` | Biến môi trường chưa set | Chạy `setx SUMO_HOME "..."` rồi mở lại terminal |
| `traci.exceptions.FatalTraCIError` | SUMO crash hoặc port bị chiếm | Đóng mọi cửa sổ SUMO đang mở, chạy lại |
| `ModuleNotFoundError: No module named 'traci'` | traci chưa cài trong môi trường | `pip install traci` hoặc thêm `%SUMO_HOME%\tools` vào `PYTHONPATH` |

---

## 5. Các điểm nhấn kỹ thuật

### 5.1. Không gian tìm kiếm hiệu dụng 4 chiều

Không gian Phenotype $\mathbb{Z}^6$ bị thu hẹp xuống **đa tạp affine $D_{\text{eff}} = 4$ chiều** bởi 2 ràng buộc đẳng thức độc lập tuyến tính. Đây không phải chi tiết cài đặt mà là **tính chất toán học bất biến** với mọi mạng lưới $N$ nút:

$$D_{\text{eff}} = \sum_{i=1}^{N} P_i$$

Với $N=2$, $P_1=P_2=2$: $D_{\text{eff}} = 4$.  
Tính mở rộng (scalability): thêm 1 nút $P_{N+1}$ pha chỉ tăng thêm đúng $P_{N+1}$ chiều.

NSGA-II tìm kiếm trên Genotype $\mathbf{u} = [C, g_{1,1}, g_{2,1}, \theta_2]^T \in \mathbb{R}^4$ — một **khối hộp vuông vức** (axis-aligned hypercube) — điều kiện để SBX và PM hoạt động đúng với phân bố xác suất thiết kế.

### 5.2. Xử lý trùng lặp trong không gian rời rạc

Trong $\mathbb{R}^4$, xác suất hai điểm lấy mẫu trùng nhau bằng 0 theo độ đo Lebesgue. Tuy nhiên, ánh xạ giải mã $\mathcal{M}$ là **hàm bước** (step function): toàn bộ vùng liên tục trong $\mathbb{R}^4$ ánh xạ về cùng một điểm nguyên trong $\mathbb{Z}^6$. Theo nguyên lý chuồng bồ câu, đụng độ (collision) là không thể tránh khi quần thể đủ lớn.

Giải pháp: `RepairedSampling` và `NSGA2WithDiversityInfill` kiểm tra tính duy nhất bằng bảng băm trên Phenotype số nguyên với chi phí $\mathcal{O}(1)$ mỗi tra cứu.

### 5.0. Luồng thực thi tổng quan

```
run_nsga2_v2.py
      │
      ├─ [init] RepairedSampling ──────────────────┐
      │          sinh N cá thể unique + khả thi    │
      │                                            ▼
      │                                   repair_operator_ver2.py
      │                                   _is_feasible() + _repair()
      │
      ├─ [loop] NSGA2WithDiversityInfill.next()
      │          │
      │          ├─ SBX crossover + PM mutation
      │          ├─ _infill(): phát hiện trùng → mutation retry → random fill
      │          └─ problem_v2_honest._evaluate()
      │                   │
      │                   └─ SUMOEvaluatorImproved.evaluate(C,g1_1,g1_2,g2_1,g2_2,O2)
      │                            │
      │                            ├─ traci.start() → simulationStep() × 900
      │                            ├─ _compute_f1_control_delay(summary.xml)   → f1
      │                            ├─ _compute_co2(emission.xml)               → f2
      │                            ├─ _compute_gridlock_cv(summary.xml)        → G
      │                            └─ _compute_f1_timeloss_unfinished(tripinfo) → validation
      │
      ├─ [per gen] RunningNormalizer.update(F_valid)
      │            HVEarlyStopping.update(F_front, normalizer)
      │            record_and_print_gen()
      │
      └─ [end] plot_results() + save CSV
```

---

### 5.3. Hypervolume với hệ quy chiếu động

**Vấn đề:** `RunningNormalizer` mở rộng khi tìm được kỷ lục mới làm tọa độ chuẩn hóa của front cũ bị dịch chuyển → $HV_{\text{best}}$ tính lại bị giảm giả tạo → early stopping kích hoạt sai.

**Giải pháp trong `HVEarlyStopping`:** Lưu **tọa độ gốc chưa chuẩn hóa** (`F_raw`) của front tốt nhất. Mỗi khi so sánh, cả front hiện tại lẫn `F_raw` đều được chuẩn hóa lại bằng normalizer **hiện tại** trước khi tính $HV$, đảm bảo cùng hệ quy chiếu.

**Ref point cho báo cáo và `extract_results.py` / `plot_pareto.py`:** Dùng ref point cố định $(1.1, 1.1)^T$ trên không gian chuẩn hóa Min-Max $[8.0, 120.0] \times [0.20, 0.45]$. Offset $\delta = 0.1$ đảm bảo nghiệm cực biên có $\tilde{f}_k = 1.0$ vẫn đóng góp diện tích $\delta$ vào $\lambda_2$, bảo toàn đầy đủ độ đo Lebesgue.
