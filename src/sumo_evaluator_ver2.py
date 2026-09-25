# -*- coding: utf-8 -*-
import os
import time
import traci
import xml.etree.ElementTree as ET
from typing import Tuple
 
class SUMOEvaluatorImproved:
    """
    Đánh giá một cá thể bằng mô phỏng SUMO/TraCI.

    Cải tiến:
    - f1 = d_control ≈ meanWaitingTime / KAPPA  [s/xe]
      KAPPA = 0.9 (hằng số vật lý, TRB E-C014 2001) — KHÔNG đổi theo kịch bản.
      Lý do: kappa phụ thuộc động học xe (decel/accel), không phụ thuộc mật độ lưu lượng.
      Thay đổi kappa theo kịch bản sẽ phá vỡ tính so sánh cross-scenario.
    - f2 = CO2 trung bình theo nhu cầu lý thuyết  [kg/xe]
    - Retry mechanism × max_retries
    - Circuit breaker: >= max_retries lỗi liên tiếp → dừng có kiểm soát
    - Logging chi tiết ra file và console
    """

    # Hệ số quy đổi stopped delay → control delay (TRB E-C014, 2001)
    # d_control ≈ d_stopped / KAPPA,  KAPPA ∈ [0.85, 0.95]
    # Trung điểm 0.9 — dùng nhất quán cho cả S1/S2/S3
    KAPPA = 0.9

    def __init__(self, sumo_config_path, sim_duration=900, warmup_time=300,
                 total_hourly_demand=2000, max_retries=3, scenario_id: str = "S2"):
        if 'SUMO_HOME' not in os.environ:
            raise EnvironmentError("Vui lòng khai báo biến môi trường SUMO_HOME")

        import sumolib
        self.sumo_binary = sumolib.checkBinary("sumo")
        self.sumo_config = sumo_config_path
        self.sim_duration = sim_duration
        self.warmup_time = warmup_time
        self.total_hourly_demand = total_hourly_demand
        self.AMBER_DURATION = 3
        self.max_retries = max_retries

        # Tracking statistics
        self.eval_count = 0
        self.error_count = 0
        self.consecutive_errors = 0

        self.scenario_id = scenario_id
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "sumo_model", "output", scenario_id)
        os.makedirs(self.output_dir, exist_ok=True)

        self.tripinfo_path = os.path.join(self.output_dir, "tripinfo.xml")
        self.summary_path  = os.path.join(self.output_dir, "summary.xml")
        self.emission_path = os.path.join(self.output_dir, "emission.xml")

        # Log file
        self.log_file = os.path.join(self.output_dir, f"sumo_eval_{scenario_id}.log")

        # Ghi log khởi tạo để xác nhận thông số kịch bản không bị sai lệch
        self._log(
            f"[INIT] scenario={scenario_id} | "
            f"total_hourly_demand={total_hourly_demand} xe/h | "
            f"sim_duration={sim_duration}s | warmup={warmup_time}s | "
            f"KAPPA={self.KAPPA} | output={self.output_dir}",
            "INFO"
        )
 
    def _log(self, message: str, level: str = "INFO"):
        """Ghi log vào file và console"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
 
    def evaluate(self, C: int, g1_1: int, g1_2: int, g2_1: int, g2_2: int, O2: int,
                 retry: int = 0) -> Tuple[float, float, float, float]:
        """
        Đánh giá một cá thể với retry mechanism
        
        Args:
            C, g1_1, g1_2, g2_1, g2_2, O2: Biến quyết định
            retry: Lần retry hiện tại (tối đa = max_retries)
        
        Returns:
            Tuple[float, float, float, float] theo thứ tự:
            - [0] f1_halting    [s/xe] : d_control = Σhalting·Δt/N_demand/KAPPA
                                         (halting-based queue integral — DÙNG CHO TỐI ƯU)
            - [1] f2_co2        [kg/xe]: phát thải CO2 / N_demand
            - [2] cv_gridlock          : ràng buộc gridlock (>0 vi phạm, ≤0 khả thi)
            - [3] f1_timeloss_val[s/xe]: timeLoss/N_demand từ tripinfo — VALIDATION ONLY
                                         (có survivorship bias, KHÔNG dùng làm f1)
        """
        self.eval_count += 1
        
        try:
            # Validate input
            if not self._validate_bounds(C, g1_1, g1_2, g2_1, g2_2, O2):
                self._log(f"Invalid bounds: C={C}, g1=({g1_1},{g1_2}), g2=({g2_1},{g2_2}), O2={O2}", "WARN")
                return 99999.0, 99999.0, 99999.0, 99999.0
            
            # Kiểm tra ràng buộc
            cv1 = g1_1 + g1_2 + 8 - C
            cv2 = g2_1 + g2_2 + 8 - C
            if abs(cv1) > 0.01 or abs(cv2) > 0.01:
                self._log(f"Constraint violation: h1={cv1}, h2={cv2}", "WARN")
                return 99999.0, 99999.0, 99999.0, 99999.0
            
            # Chạy SUMO
            sumo_cmd = [
                self.sumo_binary, "-c", self.sumo_config,
                "--tripinfo-output", self.tripinfo_path,
                "--tripinfo-output.write-unfinished", "true",  # FIXED: ghi xe chua thoat
                "--summary-output", self.summary_path,
                "--emission-output", self.emission_path,
                "--no-warnings", "true", "--no-step-log", "true"
            ]
            
            # Xóa file cũ
            for path in [self.tripinfo_path, self.summary_path, self.emission_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            traci.start(sumo_cmd)
            self._setup_timing_plans(C, g1_1, g1_2, g2_1, g2_2, O2)
            
            while float(traci.simulation.getTime()) < self.sim_duration:
                traci.simulationStep()
                
        except Exception as e:
            self.error_count += 1
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= self.max_retries:
                self._log(f"SUMO failed {self.consecutive_errors} times. Terminating.", "ERROR")
                raise RuntimeError("SUMO simulator crashed too many times")
            
            if retry < self.max_retries:
                self._log(f"Retry {retry+1}/{self.max_retries}: {str(e)}", "WARN")
                time.sleep(0.5)
                return self.evaluate(C, g1_1, g1_2, g2_1, g2_2, O2, retry=retry+1)
            else:
                self._log(f"Evaluation failed after {self.max_retries} retries: {str(e)}", "ERROR")
                return 99999.0, 99999.0, 99999.0, 99999.0
                
        finally:
            try:
                traci.close()
            except:
                pass
            time.sleep(0.3)
        
        # Trích xuất dữ liệu
        try:
            # INDEX 0 — f1 CHÍNH: halting-based queue integral từ summary.xml
            # d_control = Σhalting·Δt / N_demand / KAPPA   [s/xe]
            # Không có survivorship bias; mẫu số N_demand là hằng số
            f1_halting   = self._compute_f1_control_delay(self.summary_path)
            # INDEX 1 — f2: phát thải CO2 trung bình [kg/xe]
            f2_co2       = self._compute_co2(self.emission_path)
            # INDEX 2 — constraint: gridlock (>0 vi phạm)
            cv_gridlock  = self._compute_gridlock_cv(self.summary_path)
            # INDEX 3 — validation only: timeLoss từ tripinfo — có survivorship bias
            # KHÔNG dùng làm f1 trong NSGA-II; chỉ dùng để so sánh sau tối ưu
            f1_timeloss_val, num_veh = self._compute_f1_timeloss_unfinished(self.tripinfo_path)

            # Reset consecutive errors nếu thành công
            self.consecutive_errors = 0

            # Log thành công
            if self.eval_count % 10 == 0:
                self._log(
                    f"Eval #{self.eval_count}: "
                    f"f1_halting={f1_halting:.2f}s/veh | "
                    f"f2_co2={f2_co2:.4f}kg/veh | "
                    f"cv_gridlock={cv_gridlock:.4f} | "
                    f"f1_timeloss_val={f1_timeloss_val:.2f}s/veh (n={num_veh})",
                    "DEBUG"
                )

            # Thứ tự tuple: (f1_halting, f2_co2, cv_gridlock, f1_timeloss_validation)
            # Index 0 → f1 tối ưu  | Index 1 → f2 tối ưu
            # Index 2 → constraint  | Index 3 → validation only
            return f1_halting, f2_co2, cv_gridlock, f1_timeloss_val

        except Exception as e:
            self._log(f"Data extraction failed: {str(e)}", "ERROR")
            return 99999.0, 99999.0, 99999.0, 99999.0
    
    def _validate_bounds(self, C: int, g1_1: int, g1_2: int, g2_1: int, g2_2: int, O2: int) -> bool:
        """Kiểm tra biến nằm trong bounds.
        O2 ∈ [0, C-1] — dynamic bound nhất quán với repair operator và _is_feasible."""
        return (60 <= C <= 120 and
                15 <= g1_1 <= 90 and 15 <= g1_2 <= 90 and
                15 <= g2_1 <= 90 and 15 <= g2_2 <= 90 and
                0 <= O2 <= C - 1)
    
    def _setup_timing_plans(self, C, g1_1, g1_2, g2_1, g2_2, O2):
        phases_j61 = [
            traci.trafficlight.Phase(g1_1, "GGgrrrGGgrrr"),
            traci.trafficlight.Phase(self.AMBER_DURATION, "yyyrrryyyrrr"),
            traci.trafficlight.Phase(g1_2, "rrrGGgrrrGGg"),
            traci.trafficlight.Phase(self.AMBER_DURATION, "rrryyyrrryyy")
        ]
        logic_j61 = traci.trafficlight.Logic("Program_J61", 0, 0, phases_j61)
        traci.trafficlight.setProgramLogic("J61", logic_j61)
        
        phases_j0 = []
        if O2 > 0:
            phases_j0.append(traci.trafficlight.Phase(O2, "rrrrrrrrrrrr"))
        phases_j0.extend([
            traci.trafficlight.Phase(g2_1, "GGgrrrGGgrrr"),
            traci.trafficlight.Phase(self.AMBER_DURATION, "yyyrrryyyrrr"),
            traci.trafficlight.Phase(g2_2, "rrrGGgrrrGGg"),
            traci.trafficlight.Phase(self.AMBER_DURATION, "rrryyyrrryyy")
        ])
        logic_j0 = traci.trafficlight.Logic("Program_J0", 0, 0, phases_j0)
        traci.trafficlight.setProgramLogic("J0", logic_j0)


    # ================= CÁC HÀM XỬ LÝ NỘI BỘ (Bổ sung đầy đủ) =================
    
    def _compute_f1_control_delay(self, summary_file: str) -> float:
        """
        Tính f1 = Tổng stopped delay toàn mạng / N_demand  [s/xe]
        bằng nguyên lý Tích phân hàng chờ (Area under the queue curve).

        === Tại sao KHÔNG dùng trung bình có trọng số running (logic cũ) ===
        Logic cũ:  Σ[meanWaitingTime(t) × running(t)] / Σ[running(t)]
        Vấn đề:   running(t) là biến phụ thuộc x — mẫu số thay đổi theo nghiệm.
                  Nghiệm có nhiều xe chạy bon bon (running lớn) bị "phạt" oan
                  vì mẫu số lớn hơn. Gây méo tỷ lệ khi so sánh các nghiệm.
                  Thêm nữa: running tương quan dương với CO2 → tăng tương quan
                  nhân tạo giữa f1 và f2, làm mờ xung đột Pareto.

        === Nguyên lý Tích phân hàng chờ (Queue Integral) ===
        Mỗi giây có 1 xe dừng (v < 0.1 m/s) → hệ thống sinh ra 1 veh·second trễ.

        Công thức:
            Total_stopped_delay = Σ_{t=Twu}^{Tsim} halting(t) · Δt   [veh·s]
            f1 = Total_stopped_delay / N_demand                        [s/veh]

        Trong đó:
            halting(t) : số xe v < 0.1 m/s tại bước t — từ summary.xml
            Δt = 1s     : bước thời gian SUMO
            N_demand    : lưu lượng lý thuyết trong [Twu, Tsim] — HẰNG SỐ
                        = total_hourly_demand × (Tsim - Twu) / 3600

        Ưu điểm:
        1. Mẫu số N_demand là hằng số — KHÔNG phụ thuộc nghiệm x.
           → Phép chia chỉ scale tuyến tính, không làm sai thứ tự so sánh.
        2. Nhất quán với f2: cả hai đều chia cho N_demand → cùng "nền" dân số xe
           → đảm bảo xung đột Pareto thực sự, không bị nhiễu bởi mẫu số khác nhau.
        3. Không có survivorship bias: halting tính tất cả xe kẹt tại từng bước,
           kể cả xe chưa bao giờ thoát mạng trong thời gian mô phỏng.
        4. Tính đơn điệu nghiêm ngặt: gridlock tăng → halting(t) tăng → f1 tăng.

        Hệ số KAPPA = 0.9:
            halting chỉ đo stopped delay (v < 0.1 m/s), bỏ sót deceleration/
            acceleration delay (~5–15%). Nhân 1/KAPPA để xấp xỉ HCM control delay.
            KHÔNG đổi theo kịch bản: KAPPA là hằng số vật lý (TRB E-C014, 2001).

        Trả về: d_control xấp xỉ [s/xe], hoặc 99999.0 nếu không đọc được dữ liệu.
        """
        try:
            tree = ET.parse(summary_file)
            root = tree.getroot()
        except Exception as e:
            self._log(f"Cannot parse summary for f1: {e}", "ERROR")
            return 99999.0

        total_halting_veh_seconds = 0.0   # Σ halting(t) · Δt  [veh·s]
        valid_steps = 0

        for step in root.findall("step"):
            t = float(step.attrib.get("time", 0))
            if t < self.warmup_time:
                continue
            try:
                h = float(step.attrib.get("halting", 0))
                total_halting_veh_seconds += h   # Δt = 1s nên không cần nhân
                valid_steps += 1
            except ValueError:
                pass

        if valid_steps == 0:
            self._log("No valid halting steps in summary — returning 99999", "WARN")
            return 99999.0

        # N_demand: lưu lượng lý thuyết trong cửa sổ [Twu, Tsim] — HẰNG SỐ
        sim_window_h = (self.sim_duration - self.warmup_time) / 3600.0
        N_demand = self.total_hourly_demand * sim_window_h   # [xe]

        if N_demand <= 0:
            self._log("N_demand = 0 — kiểm tra total_hourly_demand trong config", "ERROR")
            return 99999.0

        d_stopped = total_halting_veh_seconds / N_demand     # [s/xe] stopped delay
        d_control = d_stopped / self.KAPPA                   # [s/xe] ≈ HCM control delay

        self._log(
            f"f1: halting_integral={total_halting_veh_seconds:.0f}veh*s | "
            f"N_demand={N_demand:.1f}xe | "
            f"d_stopped={d_stopped:.2f}s/veh -> d_control={d_control:.2f}s/veh "
            f"(KAPPA={self.KAPPA})",
            "DEBUG"
        )
        return d_control

    def _compute_f1_timeloss_unfinished(self, tripinfo_file: str):
        """
        VALIDATION ONLY — KHÔNG dùng làm f1 trong NSGA-II.

        Tính timeLoss trung bình toàn mạng [s/xe] từ tripinfo.xml (kể cả xe chưa thoát).
        Chỉ dùng để so sánh cross-check với f1_halting sau tối ưu.

        Lý do KHÔNG dùng làm f1 tối ưu:
            timeLoss bao gồm decel+stop+accel delay, tương quan cao với CO2
            (cả hai đều tỷ lệ với tổng năng lượng tiêu hao) → r(f1,f2) ≈ 0.99
            → mất Pareto front có nghĩa.
            f1_halting (queue integral) độc lập hơn với f2_co2 (idle emission)
            → xung đột Pareto thực sự.

        Công thức:
            f1_val = Σ timeLoss_v / N_demand   [s/xe]
            N_demand = hằng số — nhất quán với f1_halting và f2_co2.

        Lọc: chỉ tính xe có depart >= warmup_time.

        Trả về: (f1_val [s/xe], số xe được tính)
        """
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
        except Exception as e:
            self._log(f"Cannot parse tripinfo for f1: {e}", "ERROR")
            return 99999.0, 0

        total_timeloss = 0.0
        veh_count = 0
        for trip in root.findall("tripinfo"):
            try:
                depart = float(trip.attrib.get("depart", -1))
                if depart < self.warmup_time:
                    continue
                tl = float(trip.attrib.get("timeLoss", 0))
                total_timeloss += tl
                veh_count += 1
            except ValueError:
                pass

        if veh_count == 0:
            self._log("No vehicles in tripinfo after warmup -- returning 99999", "WARN")
            return 99999.0, 0

        # Mẫu số: N_demand — hằng số theo kịch bản, KHÔNG phụ thuộc nghiệm x.
        # Dùng N_demand thay vì max(veh_count, N_demand) để đảm bảo tính nhất quán:
        # hàm này chỉ dùng cho validation, cần cùng "nền dân số" với f1_halting.
        sim_window_h = (self.sim_duration - self.warmup_time) / 3600.0
        N_demand = self.total_hourly_demand * sim_window_h
        denominator = N_demand  # hằng số, không phụ thuộc nghiệm

        f1 = total_timeloss / denominator  # [s/xe]

        self._log(
            f"f1_timeloss_val: total={total_timeloss:.0f}s | "
            f"n_veh={veh_count} | N_demand={N_demand:.1f} | f1_val={f1:.3f}s/veh",
            "DEBUG"
        )
        return f1, veh_count

    def _compute_delay_tripinfo(self, tripinfo_file: str):
        """
        Tính timeLoss trung bình từ tripinfo.xml — CHỈ DÙNG CHO VALIDATION.

        CẢNH BÁO SURVIVORSHIP BIAS:
            tripinfo.xml chỉ ghi xe đã hoàn thành hành trình (đã thoát mạng).
            Xe còn kẹt lúc T_sim kết thúc bị bỏ sót hoàn toàn.
            → Khi gridlock tăng, f1_tripinfo có thể GIẢM giả tạo (chỉ xe
              thoát nhanh được tính). Vi phạm tính đơn điệu — KHÔNG dùng
              làm hàm mục tiêu trong NSGA-II.

        Dùng để: báo cáo/validation sau khi tối ưu đã hội tụ (so sánh với
        HCM control delay đo thực địa).

        Trả về: (avg_timeloss [s/xe], số xe đã thoát)
        """
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
        except Exception:
            return 99999.0, 0

        total_delay = 0.0
        veh_count = 0
        for trip in root.findall("tripinfo"):
            depart = float(trip.attrib.get("depart", 0))
            if depart >= self.warmup_time:
                try:
                    total_delay += float(trip.attrib.get("timeLoss", 0))
                    veh_count += 1
                except ValueError:
                    pass

        avg_delay = total_delay / veh_count if veh_count > 0 else 99999.0
        return avg_delay, veh_count

    def _compute_co2(self, emission_file: str) -> float:
        """
        Tính f2 = Tổng phát thải CO2 toàn mạng / N_demand  [kg/xe]

        Công thức (Lựa chọn B — nhất quán với f1):
            f2(x) = (1/10^6) · Σ_{t=Twu}^{Tsim} Σ_{v∈V(t)} E_CO2(vv,av)·Δt / N_demand

        Nhất quán với f1:
            f1 [s/xe]  và  f2 [kg/xe] — đều chia cùng N_demand (hằng số).
            → Xung đột Pareto thực sự, không bị nhiễu bởi mẫu số khác nhau.

        Tính đơn điệu: xe kẹt tiếp tục phát thải idle (α₀>0 trong HBEFA3)
            → f2 tăng đơn điệu khi gridlock tăng.

        Trả về: f2 [kg/xe], hoặc 99999.0 nếu không đọc được dữ liệu.
        """
        try:
            tree = ET.parse(emission_file)
            root = tree.getroot()
        except Exception as e:
            self._log(f"Cannot parse emission for f2: {e}", "ERROR")
            return 99999.0

        total_co2_mg = 0.0
        for timestep in root.findall("timestep"):
            t = float(timestep.attrib.get("time", 0))
            if t < self.warmup_time:
                continue
            for veh in timestep.findall("vehicle"):
                try:
                    total_co2_mg += float(veh.attrib.get("CO2", 0))
                except ValueError:
                    pass

        # N_demand: HẰNG SỐ — nhất quán với mẫu số của f1
        sim_window_h = (self.sim_duration - self.warmup_time) / 3600.0
        N_demand = self.total_hourly_demand * sim_window_h

        if N_demand <= 0:
            self._log("N_demand = 0 trong _compute_co2 — kiểm tra config", "ERROR")
            return 99999.0

        f2 = (total_co2_mg / 1_000_000.0) / N_demand   # [kg/xe]

        self._log(
            f"f2: CO2_total={total_co2_mg/1e6:.4f}kg | N_demand={N_demand:.1f}xe | f2={f2:.6f}kg/veh",
            "DEBUG"
        )
        return f2

    def _compute_gridlock_cv(self, summary_file: str, theta_max: float = 0.6) -> float:
        """
        Tính ràng buộc bất đẳng thức: g(x) = r_halt(x) - theta_max

        Định nghĩa (Chuong456_Final.pdf, mục 4.3.2):
            r_halt(x) = Σ_{t=Twu}^{Tsim} halting(t)
                        ─────────────────────────────
                        Σ_{t=Twu}^{Tsim} running(t)

        Sửa mẫu số từ loaded → running:
            loaded(t) = tổng xe đã được nạp vào mạng từ đầu mô phỏng (tích lũy, KHÔNG giảm).
            running(t) = số xe đang thực sự lưu thông trong mạng tại bước t.
            → running(t) là mẫu số đúng để tính tỷ lệ xe dừng tại một thời điểm.
            → loaded quá lớn so với running, làm r_halt bị ước lượng thấp hơn thực tế.

        Ràng buộc:
            g(x) ≤ 0  ↔  r_halt ≤ theta_max = 0.6
            g(x) > 0  → nghiệm vi phạm, bị constrained dominance của NSGA-II loại.

        Trả về:
            max(0, r_halt - theta_max) — dương = vi phạm, 0/âm = khả thi.
            99999.0 nếu running tổng = 0 (gridlock hoàn toàn ngay từ đầu).
        """
        try:
            tree = ET.parse(summary_file)
            root = tree.getroot()
        except Exception as e:
            self._log(f"Cannot parse summary for gridlock_cv: {e}", "ERROR")
            return 99999.0

        sum_halting = 0.0
        sum_running = 0.0
        for step in root.findall("step"):
            t = float(step.attrib.get("time", 0))
            if t < self.warmup_time:
                continue
            try:
                sum_halting += float(step.attrib.get("halting", 0))
                sum_running += float(step.attrib.get("running", 0))
            except ValueError:
                pass

        if sum_running <= 0:
            self._log("sum_running = 0 — gridlock hoàn toàn, trả về 99999", "WARN")
            return 99999.0

        r_halt = sum_halting / sum_running
        cv = max(0.0, r_halt - theta_max)

        self._log(
            f"gridlock: r_halt={r_halt:.4f} | theta_max={theta_max} | cv={cv:.4f}",
            "DEBUG"
        )
        return cv