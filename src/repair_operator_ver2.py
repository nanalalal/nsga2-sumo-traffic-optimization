# -*- coding: utf-8 -*-  # FIXED L8: thêm coding declaration
import numpy as np
from pymoo.core.repair import Repair
from typing import Tuple

class TrafficSignalRepairImproved(Repair):
    def __init__(self):
        super().__init__()
        self.stats = {
            "total": 0,
            "repaired": 0,
            "failed": 0,
            "feasible": 0
        }
    
    def _do(self, problem, Z, **kwargs):
        """Hàm này được pymoo tự động gọi. Z là mảng 2D chứa các cá thể."""
        for i in range(len(Z)):
            self.stats["total"] += 1
            x = Z[i]
            C, g1_1, g1_2, g2_1, g2_2, O2 = [int(v) for v in x]
            
            try:
                # Kiểm tra đã khả thi chưa
                if self._is_feasible(C, g1_1, g1_2, g2_1, g2_2, O2):
                    self.stats["feasible"] += 1
                    continue  # Bỏ qua, giữ nguyên Z[i]
                
                # Tiến hành sửa chữa
                C_new, g1_1_new, g1_2_new, g2_1_new, g2_2_new, O2_new = self._repair(
                    C, g1_1, g1_2, g2_1, g2_2, O2
                )
                
                # Double-check sau khi sửa
                if not self._is_feasible(C_new, g1_1_new, g1_2_new, g2_1_new, g2_2_new, O2_new):
                    self.stats["failed"] += 1
                    # Fallback an toàn nếu sửa thất bại để không crash SUMO
                    # FIXED L4: O2_new phải recompute cho C_fallback=64, không dùng % C_original
                    O2_fallback = int(O2_new) % 64
                    Z[i] = [64.0, 20.0, 36.0, 20.0, 36.0, float(O2_fallback)]
                else:
                    self.stats["repaired"] += 1
                    Z[i] = [float(C_new), float(g1_1_new), float(g1_2_new), 
                           float(g2_1_new), float(g2_2_new), float(O2_new)]
            except Exception as e:
                print(f"[REPAIR ERROR] {str(e)}")
                self.stats["failed"] += 1
                
        return Z
    
    def _is_feasible(self, C: int, g1_1: int, g1_2: int, g2_1: int, g2_2: int, O2: int) -> bool:
        h1 = g1_1 + g1_2 + 8 - C
        h2 = g2_1 + g2_2 + 8 - C
        
        if h1 != 0 or h2 != 0: return False
        if not (60 <= C <= 120): return False
        if not (15 <= g1_1 <= 90 and 15 <= g1_2 <= 90): return False
        if not (15 <= g2_1 <= 90 and 15 <= g2_2 <= 90): return False
        if not (0 <= O2 <= C - 1): return False
        return True
    
    def _repair(self, C: int, g1_1: int, g1_2: int, g2_1: int, g2_2: int, O2: int) -> Tuple:
        delta1 = C - (g1_1 + g1_2 + 8)
        delta2 = C - (g2_1 + g2_2 + 8)
        
        g1_1_new, g1_2_new = self._repair_node(g1_1, g1_2, delta1)
        g2_1_new, g2_2_new = self._repair_node(g2_1, g2_2, delta2)
        
        # SỬA LẠI DÒNG NÀY: Ép O2 nằm gọn trong chu kỳ C
        O2_new = int(O2) % int(C)
        
        return C, g1_1_new, g1_2_new, g2_1_new, g2_2_new, O2_new
    
    def _repair_node(self, g1: int, g2: int, delta: int) -> Tuple[int, int]:
        """
        Chiếu (g1, g2) về điểm khả thi gần nhất thoả mãn:
          g1 + g2 = g1 + g2 + delta  (sau điều chỉnh)
          15 <= g1, g2 <= 90
        Thuật toán: phân bổ delta cho g1 trước, phần còn lại cho g2.
        """
        g_min, g_max = 15, 90
        if delta == 0:
            return g1, g2
 
        # Ngau nhien chon thu tu uu tien: 0 = g1 truoc, 1 = g2 truoc
        if np.random.randint(2) == 0:
            g1_new  = int(np.clip(g1 + delta, g_min, g_max))
            absorbed = g1_new - g1
            g2_new  = int(np.clip(g2 + (delta - absorbed), g_min, g_max))
        else:
            g2_new  = int(np.clip(g2 + delta, g_min, g_max))
            absorbed = g2_new - g2
            g1_new  = int(np.clip(g1 + (delta - absorbed), g_min, g_max))
 
        return g1_new, g2_new
    
    def print_stats(self):
        total = self.stats["total"]
        if total == 0: return
        print("\n" + "="*60)
        print("REPAIR OPERATOR STATISTICS")
        print("="*60)
        print(f"Total Calls: {total}")
        print(f"Already Feasible: {self.stats['feasible']} ({100*self.stats['feasible']/total:.1f}%)")
        print(f"Successfully Repaired: {self.stats['repaired']} ({100*self.stats['repaired']/total:.1f}%)")
        print(f"Failed: {self.stats['failed']} ({100*self.stats['failed']/total:.1f}%)")
        print("="*60 + "\n")