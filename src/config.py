# -*- coding: utf-8 -*-  # FIXED L9: thêm coding declaration
# config.py
#
# KAPPA = 0.9 (d_stopped → d_control) được định nghĩa trong SUMOEvaluatorImproved.KAPPA
# và dùng nhất quán cho CẢ BA kịch bản — không khai báo riêng ở đây.
# Lý do: KAPPA là hằng số vật lý (động học xe), không phụ thuộc lưu lượng.

SCENARIOS = {
    "S1": {
        "total_hourly_demand": 2000,
        "label": "Duoi_bao_hoa_doi_xung",
        "sumo_cfg": r"D:\DoAn_NSGA2_SUMO\sumo_model_s1\simulation.sumocfg"
    },
    "S2": {
        "total_hourly_demand": 2000,
        "label": "Duoi_bao_hoa_bat_doi_xung",
        "sumo_cfg": r"D:\DoAn_NSGA2_SUMO\sumo_model_s2\simulation.sumocfg"
    },
    "S3": {
        "total_hourly_demand": 3600,
        "label": "Gan_bao_hoa_bat_doi_xung",
        "sumo_cfg": r"D:\DoAn_NSGA2_SUMO\sumo_model_s3\simulation.sumocfg"
    },
}