import torch
import GA
import pattern


# 定义  l波长 d阵列间距 delta采样精度 theta_0波束指向（角度制） L阵元数
theta_min = -90.0
theta_max = 90.0
(l, delta, theta_0) = (1, 360, 0)
d = l/2
G = 200
NP = 100
m = 1
n = 1
L = 40
Pc = 0.8
Pm = 0.050
dna = torch.randint(0, 2, (NP, m, n, L)).to(dtype=torch.float)


# 初始方向图
Fdb: torch.Tensor = pattern.pattern(dna[0], l, d, delta, theta_0)
pattern.plot(Fdb, delta, theta_min, theta_max)


print("开始优化")
ybest, dnabest = GA.GA(dna, L, m, n, G, Pc, Pm, l, d, delta, theta_0)
print("结束优化")


# 绘MSLL优化曲线
GA.plot(G, ybest)


# 绘优化后方向图
Fdb: torch.Tensor = pattern.pattern(dnabest, l, d, delta, theta_0)
pattern.plot(Fdb, delta, theta_min, theta_max)
