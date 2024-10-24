import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
import numpy as np
import jax.random as jr

tfd = tfp.distributions
tfb = tfp.bijectors

# 1. 将一个基础分布通过 tanh 函数进行变换，创建一个新的分布。
# 2. tanh 变换将原分布的取值范围压缩到 (-1, 1) 区间内。
# 3. 处理了变换后分布的概率计算、熵估计等问题。
# 4. 通过设置阈值来处理 tanh 在接近 ±1 时的数值不稳定性。
# 这种变换在强化学习中很常见，特别是在处理连续动作空间时。它可以将无界的动作值映射到有界区间，同时保持动作分布的概率特性。
# 这对于实现稳定的策略梯度算法很有帮助。


class TanhTransformedDistribution(tfd.TransformedDistribution):

    def __init__(self, distribution: tfd.Distribution, threshold: float = 0.999, validate_args: bool = False):
        super().__init__(distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args)
        self._threshold = threshold
        self.inverse_threshold = self.bijector.inverse(threshold)

        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p / epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = np.log(1.0 - threshold)

        self._log_prob_left = self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        self._log_prob_right = self.distribution.log_survival_function(inverse_threshold) - log_epsilon

    def log_prob(self, value, name='log_prob', **kwargs):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        value = jnp.clip(value, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            value <= -self._threshold,
            self._log_prob_left,
            jnp.where(value >= self._threshold, self._log_prob_right, super().log_prob(value)),
        )

    def entropy(self, name='entropy', **kwargs):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        seed = np.random.randint(0, 102400)
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=jr.PRNGKey(seed)), event_ndims=0
        )

    def _mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties

    @property
    def experimental_is_sharded(self):
        raise NotImplementedError

    def _sample_n(self, n, seed=None, **kwargs):
        pass

    def _variance(self, **kwargs):
        pass

    @classmethod
    def _maximum_likelihood_parameters(cls, value):
        pass
