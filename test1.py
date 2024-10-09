import jax.random as jr
import numpy as np
import einops as ei
import jax
import jax.numpy as jnp


# test_key = jr.PRNGKey(1234)
# print(f"test_key: {test_key}")
# test_keys = jr.split(test_key, 1_000)[: 5]
# print(f"test_keys: {test_keys}")
# test_keys = test_keys[0:]
# print(f"test_keys: {test_keys}")

# key_x0, _ = jr.split(test_keys[1], 2)
# print(f"key_x0: {key_x0}")
# print(f"_: {_}")

# x = jnp.linspace(0, 1, 5)
# print(x)


# state_dim = 4
# dt = 0.03
# params = {
#     "car_radius": 0.05,
#     "comm_radius": 0.5,
#     "n_rays": 32,
#     "obs_len_range": [0.1, 0.5],
#     "n_obs": 8,
#     "m": 0.1,  # mass
# }

# A = np.zeros((state_dim, state_dim), dtype=np.float32)
# A[0, 2] = 1.0
# A[1, 3] = 1.0
# A = A * dt + np.eye(state_dim)
# print(f"A: {A}")

# B = np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / params["m"], 0.0], [0.0, 1.0 / params["m"]]]) * dt
# print(f"B: {B}")

# pos = np.array([0.0, 0.0])
# all_obs_pos = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
# o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
# k_dist_sq = o_dist_sq - 4 * (1.01 * 0.01) ** 2
# print(f"o_dist_sq: {o_dist_sq}")
# X, Y, Z, PSI, THETA, PHI, U, V, W, R, Q, P = range(12)
# print(X, Y, Z, PSI, THETA, PHI, U, V, W, R, Q, P)



# 定义一个简单的标量函数
def f(x):
    return x**2 + 3*x + 2

# 计算 f 对 x 的雅可比矩阵
jacobian_f = jax.jacfwd(f)

# 在 x = 2 处评估雅可比矩阵
x = 2.0
print(jacobian_f(x))  # 输出: 7.0

# 例子 2: 向量值函数
# 定义一个向量值函数
def g(x):
    return jnp.array([x[0]**2, x[1]**3, x[0] * x[1]])

# 计算 g 对 x 的雅可比矩阵
jacobian_g = jax.jacfwd(g)

# 在 x = [1.0, 2.0] 处评估雅可比矩阵
x = jnp.array([1.0, 2.0])
print(jacobian_g(x))

# 例子 3: 多参数函数
# def h(x, y):
#     return x * y + jnp.sin(x)

# # 计算 h 对第一个参数 x 的雅可比矩阵
# jacobian_h_x = jax.jacfwd(h, argnums=0)

# # 计算 h 对第二个参数 y 的雅可比矩阵
# jacobian_h_y = jax.jacfwd(h, argnums=1)

# # 在 (x, y) = (1.0, 2.0) 处评估雅可比矩阵
# x = 1.0
# y = 2.0
# print(jacobian_h_x(x, y))  # 输出: 2.5403023
# print(jacobian_h_y(x, y))  # 输出: 1.0


# def k(x, y):
#     return jnp.array([x[0] * y[0] + jnp.sin(x[1]), x[1] * y[1] + jnp.cos(y[0])])

# # 计算 k 对第一个参数 x 的雅可比矩阵
# jacobian_k_x = jax.jacfwd(k, argnums=0)

# # 计算 k 对第二个参数 y 的雅可比矩阵
# jacobian_k_y = jax.jacfwd(k, argnums=1)

# # 在 (x, y) = ([1.0, 2.0], [3.0, 4.0]) 处评估雅可比矩阵
# x = jnp.array([1.0, 2.0])
# y = jnp.array([3.0, 4.0])
# print(jacobian_k_x(x, y))
# print(jacobian_k_y(x, y))

# import math
# print(math.cos(1))
n_node = np.zeros(5, dtype=int) 
print(n_node.ndim)


import numpy as np

# 假设的输入数据
agent_i = 3  # 代理数量
agent_j = 4  # 代理数量
k = 2        # 特征数量
nx = 5       # 状态或特征维度

# 随机生成输入张量
h_x = np.random.rand(agent_i, k, agent_j, nx)  # 形状 (3, 2, 4, 5)
print(h_x.shape)
dyn_f = np.random.rand(agent_j, nx)             # 形状 (4, 5)
print(dyn_f.shape)

# 执行求和操作
Lf_h = np.einsum('iknj,nj->ik', h_x, dyn_f)

# 输出形状
print(Lf_h.shape)  # 输出: (3, 2)