import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
fp = [f for f in font_manager.findSystemFonts() if "NanumGothic" in f or "NotoSansCJK" in f or "Noto Sans CJK" in f]
if fp:
    font_manager.fontManager.addfont(fp[0]); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp[0]).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False; plt.rcParams["axes.spines.right"] = False
COL = {"S0": "#2a78d6", "K2": "#eb6834", "S1": "#1baf7a", "K0": "#eda100", "K1": "#e87ba4", "P": "#9a9a94", "real": "#222222"}
NAME = {"P": "지속(상태 고정)", "K0": "운동학 자전거, 시뮬 지연 0.07 s, 이득 1", "K1": "운동학 자전거 + 지연 (맞춤 3개)",
        "K2": "선형 자전거(언더스티어) + 지연 (맞춤 4개)", "S0": "시뮬 명목 (전체 모델)", "S1": "시뮬 + 조향 이득·트림 맞춤"}
FIG = "/home/shchon11/F1tenth_E2E/docs/pm/audit-2026-11-28/D_figs/"
