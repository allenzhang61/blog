# RNN 算法原理与实现详解 (NumPy + SGD)

本文档详细介绍了循环神经网络 (RNN) 算法原理。该实现完全基于 NumPy，手动完成了前向传播、反向传播 (BPTT) 以及随机梯度下降 (SGD) 优化过程。

## 1. 核心数学模型

### 1.1 符号定义
*   $x_t \in \mathbb{R}^{D_{in}}$: 时间步 $t$ 的输入向量。
*   $h_t \in \mathbb{R}^{D_{hidden}}$: 时间步 $t$ 的隐藏状态 (Hidden State)。
*   $y \in \mathbb{R}^{D_{out}}$: 最终输出向量 (预测值)。
*   $W_{xh} \in \mathbb{R}^{D_{in} \times D_{hidden}}$: 输入层到隐藏层的权重矩阵。
*   $W_{hh} \in \mathbb{R}^{D_{hidden} \times D_{hidden}}$: 隐藏层到隐藏层的权重矩阵 (循环权重)。
*   $W_{hy} \in \mathbb{R}^{D_{hidden} \times D_{out}}$: 隐藏层到输出层的权重矩阵。
*   $b_h \in \mathbb{R}^{D_{hidden}}$: 隐藏层偏置。
*   $b_y \in \mathbb{R}^{D_{out}}$: 输出层偏置。

### 1.2 前向传播 (Forward Propagation)
对于长度为 $T$ 的输入序列 $x_1, x_2, ..., x_T$：

**Step 1: 循环更新隐藏状态**
对于每个时间步 $t = 1$ 到 $T$：
$$
h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)
$$
*(注：$h_0$ 通常初始化为全零向量)*

**Step 2: 计算最终输出**
使用最后一个时间步的隐藏状态 $h_T$ 进行预测：
$$
y_{pred} = h_T W_{hy} + b_y
$$

### 1.3 损失函数 (Loss Function)
使用均方误差 (MSE) 衡量预测值与真实值 $y_{true}$ 之间的差距：
$$
L = \frac{1}{2N} \sum_{i=1}^{N} ||y_{pred}^{(i)} - y_{true}^{(i)}||^2
$$

### 1.4 反向传播 (BPTT - Backpropagation Through Time)
我们需要计算损失函数 $L$ 对所有参数的梯度。由于 RNN 存在时间依赖，梯度需要沿时间轴反向传播。

**1. 输出层梯度**
$$
\frac{\partial L}{\partial y_{pred}} = \frac{1}{N} (y_{pred} - y_{true})
$$
$$
\frac{\partial L}{\partial W_{hy}} = h_T^T (\frac{\partial L}{\partial y_{pred}})
$$
$$
\frac{\partial L}{\partial b_y} = \sum (\frac{\partial L}{\partial y_{pred}})
$$

**2. 隐藏层梯度 (时间反向传播)**
令 $dh_t = \frac{\partial L}{\partial h_t}$。
对于最后一个时间步 $T$：
$$
dh_T = (\frac{\partial L}{\partial y_{pred}}) W_{hy}^T
$$

对于之前的每一个时间步 $t = T, T-1, ..., 1$：
首先计算经过激活函数之前的梯度 $dh_{raw}$ (因为 $h_t = \tanh(h_{raw})$)：
$$
dh_{raw} = dh_t \odot (1 - h_t^2)
$$
*(注：$\odot$ 表示逐元素相乘，$(1 - h_t^2)$ 是 tanh 的导数)*

然后累加梯度：
$$
\frac{\partial L}{\partial W_{hh}} \leftarrow \frac{\partial L}{\partial W_{hh}} + h_{t-1}^T dh_{raw}
$$
$$
\frac{\partial L}{\partial W_{xh}} \leftarrow \frac{\partial L}{\partial W_{xh}} + x_t^T dh_{raw}
$$
$$
\frac{\partial L}{\partial b_h} \leftarrow \frac{\partial L}{\partial b_h} + \sum dh_{raw}
$$

最后计算传递给上一时刻的梯度 $dh_{t-1}$：
$$
dh_{t-1} = dh_{raw} W_{hh}^T
$$

### 1.5 参数更新 (SGD)
使用简单的随机梯度下降：
$$
\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}
$$
其中 $\eta$ 是学习率 (learning rate)。

---

## 2. 算法伪代码 (Pseudocode)

```python
Algorithm RNN_Training_SGD:
    Initialize parameters W_xh, W_hh, W_hy, b_h, b_y randomly
    Set learning_rate, num_epochs

    FOR epoch = 1 TO num_epochs:
        
        # --- 1. 前向传播 (Forward) ---
        Initialize h_states = [zeros]
        FOR t = 1 TO sequence_length:
            h_raw = x[t] @ W_xh + h_states[t-1] @ W_hh + b_h
            h_t = tanh(h_raw)
            Store h_t in h_states
        END FOR
        
        y_pred = h_states[last] @ W_hy + b_y
        
        # --- 2. 计算损失 (Loss) ---
        Loss = MSE(y_pred, y_true)
        dy = (y_pred - y_true) / batch_size  # Loss 对输出的梯度
        
        # --- 3. 反向传播 (Backward) ---
        # 3.1 输出层梯度
        dW_hy = h_states[last].T @ dy
        db_y = sum(dy)
        dh = dy @ W_hy.T  # 传递给隐藏层的初始梯度
        
        # 3.2 时间反向传播 (BPTT)
        Initialize dW_hh, dW_xh, db_h as zeros
        FOR t = sequence_length DOWNTO 1:
            # tanh 的导数: 1 - h^2
            dh_raw = dh * (1 - h_states[t]^2)
            
            # 累加当前时间步的梯度贡献
            dW_hh += h_states[t-1].T @ dh_raw
            dW_xh += x[t].T @ dh_raw
            db_h += sum(dh_raw)
            
            # 传递梯度给上一时刻
            dh = dh_raw @ W_hh.T
        END FOR
        
        # 3.3 梯度裁剪 (Gradient Clipping) - 可选但推荐
        Clip gradients to range [-1, 1] to prevent explosion
        
        # --- 4. 参数更新 (SGD Update) ---
        W_xh -= learning_rate * dW_xh
        W_hh -= learning_rate * dW_hh
        W_hy -= learning_rate * dW_hy
        b_h  -= learning_rate * db_h
        b_y  -= learning_rate * db_y
        
    END FOR
```


## 代码

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 1. 生成正弦波数据
def generate_data(seq_length=1000):
    """
    生成一个简单的正弦波序列
    """
    x = np.linspace(0, 100, seq_length)
    y = np.sin(x)
    return y

# 2. 数据预处理
def create_sequences(data, sub_seq_length):
    """
    将时间序列数据转换为监督学习问题 (X, y)
    输入序列长度为 seq_length，预测下一个点
    """
    xs = []
    ys = []
    for i in range(len(data) - sub_seq_length):
        x = data[i:(i + sub_seq_length)]
        y = data[i + sub_seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 3. 定义纯 NumPy RNN 模型
class SimpleRNN_NumPy:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重 (Xavier/Glorot Initialization)
        # W_xh: 输入 -> 隐藏层权重 (input_size, hidden_size)
        self.W_xh = np.random.randn(input_size, hidden_size) / np.sqrt(input_size) # 1x32
        # W_hh: 隐藏层 -> 隐藏层权重 (hidden_size, hidden_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size) # 32x32
        # W_hy: 隐藏层 -> 输出层权重 (hidden_size, output_size)
        self.W_hy = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size) # 32x1
        
        # 初始化偏置
        self.b_h = np.zeros((1, hidden_size)) # 隐藏层偏置 1x32
        self.b_y = np.zeros((1, output_size)) # 输出层偏置 1x1

    def forward(self, inputs):
        """
        前向传播
        inputs: (batch_size, seq_length, input_size)
        返回: 
        - output: 最终预测值 (batch_size, output_size)
        - cache: 用于反向传播的中间变量 (h_states, inputs)
        """
        batch_size, sub_seq_length, _ = inputs.shape
        
        # 保存每个时间步的隐藏状态，用于反向传播
        # h_states 维度: (batch_size, seq_length + 1, hidden_size)
        # h_states[:, -1, :] 是初始隐藏状态 (通常全0)
        h_states = np.zeros((batch_size, sub_seq_length + 1, self.hidden_size))
        
        # 时间步循环
        for t in range(sub_seq_length):
            x_t = inputs[:, t, :] # (batch_size, input_size)
            h_prev = h_states[:, t-1, :] # 上一时刻隐藏状态
            
            # RNN 公式: h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
            h_raw = np.dot(x_t, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
            h_next = np.tanh(h_raw)
            
            # 保存当前隐藏状态
            h_states[:, t, :] = h_next
            
        # 取最后一个时间步的隐藏状态进行预测
        h_last = h_states[:, sub_seq_length-1, :]
        
        # 输出层: y = h_last @ W_hy + b_y
        output = np.dot(h_last, self.W_hy) + self.b_y
        
        cache = (inputs, h_states)
        return output, cache

    def backward(self, cache, dy, learning_rate=0.01, clip_value=1.0):
        """
        反向传播 (BPTT: Backpropagation Through Time)
        cache: forward 返回的中间变量
        dy: 输出层的梯度 dL/dy (batch_size, output_size)
        """
        inputs, h_states = cache
        batch_size, seq_length, _ = inputs.shape
        
        # 初始化梯度
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        dW_hh = np.zeros_like(self.W_hh)
        dW_xh = np.zeros_like(self.W_xh)
        db_h = np.zeros_like(self.b_h)
        
        # 1. 输出层梯度
        # output = h_last @ W_hy + b_y
        # dL/dW_hy = h_last.T @ dy
        h_last = h_states[:, seq_length-1, :]
        dW_hy = np.dot(h_last.T, dy)
        db_y = np.sum(dy, axis=0, keepdims=True)
        
        # dL/dh_last = dy @ W_hy.T
        dh = np.dot(dy, self.W_hy.T)
        
        # 2. BPTT: 时间步反向传播
        for t in reversed(range(seq_length)):
            # 取出当前时刻需要的变量
            h_curr = h_states[:, t, :]
            h_prev = h_states[:, t-1, :]
            x_t = inputs[:, t, :]
            
            # h_t = tanh(h_raw)
            # dL/dh_raw = dL/dh * (1 - tanh^2(h_raw)) = dL/dh * (1 - h_curr^2)
            dh_raw = dh * (1 - h_curr ** 2)
            
            # 更新参数梯度
            # h_raw = x_t @ W_xh + h_prev @ W_hh + b_h
            dW_hh += np.dot(h_prev.T, dh_raw)
            dW_xh += np.dot(x_t.T, dh_raw)
            db_h += np.sum(dh_raw, axis=0, keepdims=True)
            
            # 计算传递给上一时刻隐藏状态的梯度
            # dL/dh_{t-1} = dL/dh_raw @ W_hh.T
            dh = np.dot(dh_raw, self.W_hh.T)
            
        # 梯度裁剪 (防止梯度爆炸)
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -clip_value, clip_value, out=dparam)
            
        # 3. 更新参数 (SGD)
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

def mse_loss(y_pred, y_true):
    """均方误差损失函数"""
    return 0.5 * np.mean((y_pred - y_true) ** 2)

def mse_loss_grad(y_pred, y_true):
    """均方误差损失函数的梯度 dL/dy"""
    return (y_pred - y_true) / y_true.shape[0]

def main():
    # --- 超参数设置 ---
    sub_seq_length = 20      # 输入序列长度
    input_size = 1       # 输入特征维度
    hidden_size = 32     # 隐藏层大小
    output_size = 1      # 输出维度
    learning_rate = 1e-3 # 学习率 (纯 NumPy SGD 可能需要较小学习率)
    num_epochs = 100     # 训练轮数

    # --- 准备数据 ---
    data = generate_data(seq_length=1000)
    X, y = create_sequences(data, sub_seq_length)  # X: 980x20, y: 980x1

    # 划分训练集和测试集
    train_size = int(len(y) * 0.8) # 784
    
    # 增加特征维度 (N, L) -> (N, L, 1)
    X_train = X[:train_size].reshape(-1, sub_seq_length, 1)
    y_train = y[:train_size].reshape(-1, 1)
    
    X_test = X[train_size:].reshape(-1, sub_seq_length, 1)
    y_test = y[train_size:].reshape(-1, 1)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- 初始化模型 ---
    model = SimpleRNN_NumPy(input_size, hidden_size, output_size)
    
    # --- 训练模型 ---
    loss_history = []
    print("Start Training...")
    
    for epoch in range(num_epochs):
        # 1. 前向传播
        outputs, cache = model.forward(X_train)
        
        # 2. 计算损失
        loss = mse_loss(outputs, y_train)
        loss_history.append(loss)
        
        # 3. 计算输出层梯度
        dy = mse_loss_grad(outputs, y_train)
        
        # 4. 反向传播并更新权重
        model.backward(cache, dy, learning_rate=learning_rate)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.6f}')

    # --- 预测与评估 ---
    train_predict, _ = model.forward(X_train)
    test_predict, _ = model.forward(X_test)

    # --- 可视化结果 ---
    plt.figure(figsize=(10, 6))
    
    # 绘制训练集
    plt.subplot(2, 1, 1)
    plt.plot(y_train, label='True Value')
    plt.plot(train_predict, label='Prediction')
    plt.title('Training Set Prediction (NumPy Only)')
    plt.legend()
    
    # 绘制测试集
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='True Value')
    plt.plot(test_predict, label='Prediction')
    plt.title('Test Set Prediction (NumPy Only)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
```