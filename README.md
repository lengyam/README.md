计算机·电气工程·应用数学的个人洞见

![Banner](https://via.placeholder.com/800x200?text=CS+EE+AppliedMath)  
*探索技术交叉点的思维实验场*

## 🌟 核心理念
> "真正的创新发生在学科边界——当电路遇见算法，当微积分驱动机器学习，当物理直觉重塑数据结构"

### 核心观点
1. **计算本质论**  
   所有复杂系统皆可抽象为状态机 + 能量约束
2. **电磁统一观**  
   麦克斯韦方程组是宇宙的API文档
3. **数学工具链**  
   微分方程 ⇄ 线性代数 ⇄ 概率统计 ⇄ 优化理论

---

## 📚 知识图谱

mermaid

graph LR

A[计算机科学] --> B(算法设计)

A --> C(系统架构)

D[电气工程] --> E(电磁场分析)

D --> F(信号处理)

G[应用数学] --> H(微分方程)

G --> I(随机过程)

B --> J{交叉点}

E --> J

H --> J


---

## ⚡ 电气工程洞见：RLC电路与阻尼振荡
**物理本质**：能量在电容电场与电感磁场间的周期性转移

python

import numpy as np

import matplotlib.pyplot as plt

def rlc_response(R=10, L=0.1, C=100e-6, V0=5, t_max=0.1):

"""求解RLC串联电路微分方程"""

w0 = 1/np.sqrt(L*C)          # 固有频率

alpha = R/(2*L)              # 阻尼系数

if alpha < w0:               # 欠阻尼
    wd = np.sqrt(w0**2 - alpha**2)
    def v(t): return V0 * np.exp(-alpha*t) * np.cos(wd*t)
else:                        # 过阻尼
    beta = np.sqrt(alpha**2 - w0**2)
    def v(t): return V0 * (np.exp((-alpha+beta)*t) + np.exp((-alpha-beta)*t))/2

t = np.linspace(0, t_max, 1000)
return t, [v(x) for x in t]

t, v = rlc_response()

plt.plot(t, v)

plt.title('RLC Circuit Response\nα=50 < ω₀=316 → Underdamped')

plt.xlabel('Time (s)'); plt.ylabel('Voltage (V)')

plt.grid(True)

**关键认知**：阻尼比ζ=α/ω₀决定系统行为，如同控制系统中的稳定性判据

---

## 💻 计算机科学洞见：FFT的物理意义
**核心思想**：将时域信号分解为旋转矢量的合成

python

import numpy as np

import matplotlib.pyplot as plt

生成复合信号：50Hz正弦波 + 120Hz余弦波

t = np.linspace(0, 1, 1000, endpoint=False)

signal = 0.7np.sin(2np.pi50t) + np.cos(2np.pi120*t)

FFT分析

fft_out = np.fft.rfft(signal)

freqs = np.fft.rfftfreq(len(t), d=t[1]-t[0])

可视化

plt.figure(figsize=(12,4))

plt.subplot(121).plot(t, signal); plt.title("Time Domain")

plt.subplot(122).stem(freqs, np.abs(fft_out)/500); plt.title("Frequency Domain")

**顿悟时刻**：FFT本质是基向量(e^(iωt))的正交分解，类似傅里叶级数但更高效

---

## ∫ 应用数学洞见：微分方程与神经网络
**统一视角**：ResNet的残差连接 ≈ 欧拉法求解常微分方程

python

import torch

import torch.nn as nn

class ODE_Net(nn.Module):

"""常微分方程的神经网络近似"""

def init(self, hidden_dim=64):

super().init()

self.net = nn.Sequential(

nn.Linear(1, hidden_dim),

nn.Tanh(),

nn.Linear(hidden_dim, 1)

)

def forward(self, x):
    # 欧拉法离散化：y' = f(x,y) ≈ (f(x+h)-f(x))/h
    h = 0.01
    y = torch.zeros_like(x)
    for i in range(len(x)-1):
        y[i+1] = y[i] + h * self.net(torch.tensor([y[i]]))
    return y

测试：近似 dy/dx = -2y

model = ODE_Net()

optimizer = torch.optim.Adam(model.parameters())

loss_fn = nn.MSELoss()

for epoch in range(1000):

x = torch.rand(100, 1) * 5

y_true = torch.exp(-2*x)

y_pred = model(x)

loss = loss_fn(y_pred, y_true)

optimizer.zero_grad()
loss.backward()
optimizer.step()

**深层联系**：神经网络的深度⇔时间步长，激活函数⇔微分算子

---

## 🧠 学习哲学
1. **三遍学习法**：
   - 第一遍：建立直觉（动画/实物演示）
   - 第二遍：形式化推导（数学证明）
   - 第三遍：工程实现（代码复现）

2. **费曼技巧变体**：

mermaid

graph TB

A[选择一个概念] --> B(向橡胶鸭解释)

B --> C{是否卡壳？}

C -->|Yes| D[回溯知识断层]

C -->|No| E[简化表述]

D --> B

E --> F[对比权威资料]


---

## 🚀 项目路线图
| 领域         | Q3目标                  | Q4目标                  |
|--------------|-------------------------|-------------------------|
| 计算机       | 实现量子电路模拟器      | 构建RISC-V教学CPU       |
| 电气工程     | 设计PCB天线阵列         | 开发EM仿真加速算法      |
| 应用数学     | 完成PDE求解器原型       | 研究拓扑数据分析应用    |

---
