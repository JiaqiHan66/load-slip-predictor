# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:58:22 2025

@author: hanjiaqi
"""


import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 全局字号
plt.rcParams['font.size'] = 12

# 坐标轴标题字号
plt.rcParams['axes.labelsize'] = 14

# 坐标轴刻度字号
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 图例字号
plt.rcParams['legend.fontsize'] = 12

# 线宽、刻度方向
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 图像边距优化
plt.rcParams['figure.autolayout'] = True

# 背景纯白
plt.rcParams['savefig.facecolor'] = 'white'

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ===== 一级参数 =====
params = {
    "s1": 2.56,
    "s2": 4.9,
    "s3": 6.67,
    "tau1": 2.3,
    "tau2": 1.45,
    "tau3": 0.414,
    "Ep": 200000,
    "r0": 7.63,
    "l": 5000
}

pi = np.pi

# ===== 二级参数 =====
k1 = params["tau1"] / params["s1"]
k2 = -(params["tau2"] - params["tau1"]) / (params["s2"] - params["s1"])
k3 = -(params["tau3"] - params["tau2"]) / (params["s3"] - params["s2"])
lam2 = (2 / params["r0"]) / params["Ep"]
alpha1 = np.sqrt(lam2 * k1)
alpha2 = np.sqrt(lam2 * k2)
alpha3 = np.sqrt(lam2 * k3)
alpha4_sq = lam2 * params["tau3"]
a = k2 * params["s1"] + params["tau1"]
b = k3 * params["s2"] + params["tau2"]

# ===== 第三级参数（解析公式） =====
def compute_ls1max():
    s1, s2 = params["s1"], params["s2"]
    num = -alpha1 * s1
    sqrt_val = (alpha1 * s1)**2 - alpha2**2 * (s2 - s1) * ((s2 + s1) - (2 * a / k2))
    den = alpha2 * ((2 * a / k2) - s1 - s2)
    return (2 / alpha2) * np.arctan((num + np.sqrt(sqrt_val)) / den)

ls1max = compute_ls1max()

def compute_mu1():
    s1, s2, l = params["s1"], params["s2"], params["l"]
    num = alpha1 * s1 * np.tanh(alpha1 * (l - ls1max)) + alpha2 * (a / k2 - s2) * np.sin(alpha2 * ls1max)
    den = np.cos(alpha2 * ls1max)
    return num / den

mu1 = compute_mu1()

def compute_ls2max():
    s2, s3 = params["s2"], params["s3"]
    inner = mu1**2 + alpha3**2 * (s3 - s2) * ((2 * b / k3) - s2 - s3)
    den = alpha3 * ((2 * b / k3) - s2 - s3)
    return (2 / alpha3) * np.arctan((-mu1 + np.sqrt(inner)) / den)

ls2max = compute_ls2max()
l = params["l"]

# ===== 各阶段绘图 =====

# 第一阶段
P1_max = params["Ep"] * pi * params["r0"]**2 * alpha1 * params["s1"] * np.tanh(alpha1 * params["l"])
P1_N = np.linspace(0, P1_max, 200)
C1 = (1 / (params["Ep"] * pi * alpha1 * params["r0"]**2)) * (np.cosh(alpha1 * l) / np.sinh(alpha1 * l))
s1_vals = C1 * P1_N
P1_kN = P1_N / 1000

# 第二阶段
ls1_array = np.linspace(0, ls1max, 500)
P2_list, s2_list = [], []
for ls1 in ls1_array:
    P2 = pi * params["r0"]**2 * params["Ep"] * (
        (a / k2 - params["s1"]) * alpha2 * np.sin(alpha2 * ls1) +
        params["s1"] * alpha1 * np.tanh(alpha1 * (l - ls1)) * np.cos(alpha2 * ls1)
    )
    num = (P2 / (pi * params["r0"]**2 * params["Ep"])) * np.cos(alpha2 * (ls1 - l + l)) - \
          params["s1"] * alpha1 * np.tanh(alpha1 * (l - ls1)) * np.cos(alpha2 * 0)
    den = alpha2 * np.sin(alpha2 * ls1)
    s2_now = a / k2 - num / den
    P2_list.append(P2 / 1000)
    s2_list.append(s2_now)

# 第三阶段
ls2_array = np.linspace(0, ls2max, 500)
P3_list, s3_list = [], []
for ls2 in ls2_array:
    P3 = params["Ep"] * pi * params["r0"]**2 * (
        mu1 * np.cos(alpha3 * ls2) + (b / k3 - params["s2"]) * alpha3 * np.sin(alpha3 * ls2)
    )
    num = -(P3 / (params["Ep"] * pi * params["r0"]**2)) * np.cos(alpha3 * (l - l - ls2)) + mu1 * np.cos(alpha3 * 0)
    den = alpha3 * np.sin(alpha3 * ls2)
    s3_now = b / k3 + num / den
    P3_list.append(P3 / 1000)
    s3_list.append(s3_now)

# 第四阶段
lf_array = np.linspace(0, l - ls1max - ls2max, 500)
P4_list, s4_list = [], []
for lf in lf_array:
    mu2 = alpha1 * params["s1"] * np.tanh(alpha1 * (l - ls1max - ls2max - lf)) + \
          alpha2 * (a / k2 - params["s2"]) * np.sin(alpha2 * ls1max)
    mu2 /= np.cos(alpha2 * ls1max)
    P4 = params["Ep"] * pi * params["r0"]**2 * (
        (-alpha3 * (params["s3"] - b / k3) * np.sin(alpha3 * ls2max) + mu2) /
        np.cos(alpha3 * ls2max) + alpha4_sq * lf
    )
    term1 = (alpha4_sq / 2) * l**2
    term2 = (P4 / (params["Ep"] * pi * params["r0"]**2) - alpha4_sq * l) * l
    term3 = (alpha4_sq / 2) * (l**2 - lf**2)
    term4 = -(P4 / (params["Ep"] * pi * params["r0"]**2)) * (l - lf)
    s4_now = term1 + term2 + term3 + term4 + params["s3"]
    P4_list.append(P4 / 1000)
    s4_list.append(s4_now)

# 第五阶段
def tau1(x, lf):
    term1 = np.cos(alpha2 * (ls1max - l + x + ls2max + lf))
    term2 = ((a / k2 - params["s2"]) * alpha2 * np.sin(alpha2 * ls1max) / np.cos(alpha2 * ls1max))
    term3 = np.cos(alpha2 * (l - x - ls2max - lf))
    return (alpha2 * params["Ep"] * params["r0"] / 2) * term1 * term2 * term3 / np.sin(alpha2 * ls1max)

def tau2(x, lf):
    mu2 = alpha1 * params["s1"] * np.tanh(alpha1 * (l - ls1max - ls2max - lf)) + \
          alpha2 * (a / k2 - params["s2"]) * np.sin(alpha2 * ls1max)
    mu2 /= np.cos(alpha2 * ls1max)
    eta2 = (b / k3 - params["s3"]) * alpha3 * np.sin(alpha3 * ls2max) + mu2
    eta2 /= np.cos(alpha3 * ls2max)
    return (alpha3 * params["Ep"] * params["r0"] / 2) * (eta2 * np.cos(alpha3 * (l - x - ls2max - lf)) -
                                                         mu2 * np.cos(alpha3 * (l - x - lf))) / np.sin(alpha3 * ls2max)

def s_lf(P, lf):
    term1 = (alpha4_sq / 2) * l**2
    term2 = (P / (params["Ep"] * pi * params["r0"]**2) - alpha4_sq * l) * l
    term3 = (l - lf) * ((alpha4_sq / 2) * (l + lf) - P / (params["Ep"] * pi * params["r0"]**2))
    return term1 + term2 + term3 + params["s3"]

lf_vals_5 = np.linspace(l - ls1max - ls2max, l - ls2max, 200)
P5_list, s5_list = [], []
for lf in lf_vals_5:
    x1, x2, x3, x4 = 0, l - ls2max - lf, l - lf, l
    integral1, _ = quad(tau1, x1, x2, args=(lf,))
    integral2, _ = quad(tau2, x2, x3, args=(lf,))
    integral3 = params["tau3"] * (x4 - x3)
    P_total = 2 * pi * params["r0"] * (integral1 + integral2 + integral3)
    s_val = s_lf(P_total, lf)
    P5_list.append(P_total / 1000)
    s5_list.append(s_val)

# 第六阶段
lf_vals_6 = np.linspace(l - ls2max, l, 200)
P6_list, s6_list = [], []
for lf in lf_vals_6:
    x1 = 0
    x2 = l - lf
    x3 = l
    integral_soft2, _ = quad(tau2, x1, x2, args=(lf,))
    integral_friction = params["tau3"] * (x3 - x2)
    P6 = 2 * pi * params["r0"] * (integral_soft2 + integral_friction)
    s6 = s_lf(P6, lf)
    P6_list.append(P6 / 1000)
    s6_list.append(s6)
# ===== 绘图 =====
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴加粗
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.figure(figsize=(12, 8))
plt.plot(s1_vals, P1_kN, label="Stage I: E", color="blue", linewidth=2)
plt.plot(s2_list, P2_list, label="Stage II: E-S1", color="darkorange", linewidth=2)
plt.plot(s3_list, P3_list, label="Stage III: E-S1-S2", color="mediumpurple", linewidth=2)
plt.plot(s4_list, P4_list, label="Stage IV: E-S1-S2-F", color="indianred", linewidth=2)
plt.plot(s5_list, P5_list, label="Stage V: S1-S2-F", color="slategray", linewidth=2)
plt.plot(s6_list, P6_list, label="Stage VI: S2-F", color="saddlebrown", linewidth=2)
plt.xlabel("Slip of the loading end $s$ (mm)", fontsize=14)
plt.ylabel("Load P (kN)", fontsize=14)
plt.title("Full-range Load-slip curve of the anchorage interface slip failure process", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
# ===== 实验数据点 =====
exp_s = [1.10186, 2.54851, 4.12026, 6.46971, 12.79475, 19.54399]
exp_P = [35.6484, 103.63597, 139.19869, 171.62353, 204.81541, 218.76158]

# ===== 重绘图并叠加实验点 =====
plt.figure(figsize=(12, 8))

# 拟合曲线（六阶段）
plt.plot(s1_vals, P1_kN, label="Stage I: E", color="blue", linewidth=2)
plt.plot(s2_list, P2_list, label="Stage II: E-S1", color="darkorange", linewidth=2)
plt.plot(s3_list, P3_list, label="Stage III: E-S1-S2", color="mediumpurple", linewidth=2)
plt.plot(s4_list, P4_list, label="Stage IV: E-S1-S2-F", color="indianred", linewidth=2)
plt.plot(s5_list, P5_list, label="Stage V: S1-S2-F", color="slategray", linewidth=2)
plt.plot(s6_list, P6_list, label="Stage VI: S2-F", color="saddlebrown", linewidth=2)

# 实验数据点（叠加）
plt.scatter(exp_s, exp_P, color="black", marker="o", s=80, label="data", zorder=10)

# ===== 图形设置 =====
plt.xlabel("Slip of the loading end $s$ (mm)", fontsize=14)
plt.ylabel("load P (kN)", fontsize=14)
plt.title("analytical vs data", fontsize=16)
plt.legend(fontsize=10,loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("fig-9-data Vs analysis.pdf", dpi=600, bbox_inches='tight')
plt.show()
# ===== 粘结-滑移关系图（τ-s 曲线） =====
import matplotlib.pyplot as plt

# 一级参数
s1 = params["s1"]
s2 = params["s2"]
s3 = params["s3"]
tau1 = params["tau1"]
tau2 = params["tau2"]
tau3 = params["tau3"]

# 分段函数：三段线性模型
s_vals_1 = np.linspace(0, s1, 100)
tau_vals_1 = (tau1 / s1) * s_vals_1  # 第一段上升

s_vals_2 = np.linspace(s1, s2, 100)
tau_vals_2 = tau1 + (tau2 - tau1) / (s2 - s1) * (s_vals_2 - s1)  # 第二段下降

s_vals_3 = np.linspace(s2, s3, 100)
tau_vals_3 = tau2 + (tau3 - tau2) / (s3 - s2) * (s_vals_3 - s2)  # 第三段下降
s4 = s3*1.5
s_vals_4 = np.linspace(s3, s4, 100)
tau_vals_4 = np.full_like(s_vals_4, tau3)
# 绘图
plt.figure(figsize=(8, 6))
plt.plot(s_vals_1, tau_vals_1, label="Stage I: Elastic", color="blue", linewidth=2)
plt.plot(s_vals_2, tau_vals_2, label="Stage II: First softening", color="darkorange", linewidth=2)
plt.plot(s_vals_3, tau_vals_3, label="Stage III: Second softening", color="mediumpurple", linewidth=2)
plt.plot(s_vals_4, tau_vals_4, label="Stage IV: Residual friction ", color="indianred", linewidth=2)

# 标注关键点
plt.scatter([s1, s2, s3], [tau1, tau2, tau3], color='black', zorder=10)
plt.text(s1, tau1, f"  ({s1:.2f}, {tau1:.2f})", fontsize=10)
plt.text(s2, tau2, f"  ({s2:.2f}, {tau2:.2f})", fontsize=10)
plt.text(s3, tau3, f"  ({s3:.2f}, {tau3:.2f})", fontsize=10)

# 设置
plt.xlabel("Slip of the loading end s (mm)", fontsize=14)
plt.ylabel(" τ (MPa)", fontsize=14)
plt.title("bond-slip model", fontsize=16)
plt.legend(fontsize=10,loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("fig-9-bondslip model.pdf", dpi=600, bbox_inches='tight')
plt.show()
