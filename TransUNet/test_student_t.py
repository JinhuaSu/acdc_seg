#%%
1
#%%
import tensorflow as tf

n1 = 10
n2 = 10
with tf.Session() as sess:
    value = tf.cast(tf.zeros([1]) - 1, tf.float32)
    t = tf.contrib.distributions.StudentT(float(n1 + n2 - 2), 0.0, 1.0)
    loss_inv = t.cdf(value)
    p_value = 1 - sess.run(loss_inv)
print(p_value)

# %%
