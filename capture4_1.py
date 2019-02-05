# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 22:14:32 2019

@author: Administrator
"""
import tensorflow as tf
sess=tf.Session()
#常量
t1 = tf.constant([4.0],tf.float32)
print(sess.run(t1))
t2 = tf.constant([4.0,3],tf.float32)
print(t2)
print(sess.run(t2))
t3 = tf.zeros([1,2],tf.float32)
print(t3)
print(sess.run(t3))
t4 = tf.ones([1,2],tf.float32)
print(t4)
print(sess.run(t4))

#创建具有不同分布的随机张量
'''
random_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
)
从正态分布中输出随机值. 
参数：

shape：一维整数张量或 Python 数组.输出张量的形状.
mean：dtype 类型的0-D张量或 Python 值.正态分布的均值.
stddev：dtype 类型的0-D张量或 Python 值.正态分布的标准差.
dtype：输出的类型.
seed：一个 Python 整数.用于为分发创建一个随机种子.查看 tf.set_random_seed 行为.
name：操作的名称(可选).
'''
t5 = tf.random_normal([1,2],mean=0.0,stddev=1 , dtype=tf.float32)
print(t5)
print(sess.run(t5))

#变量
def weight_variables(shape):
    initial = tf.random_normal(shape,mean=0,stddev=1,dtype=tf.float32)
    return tf.Variable(initial)
t6 = weight_variables([10,10])
print(t6)

def bias_variables(shape):
    initial = tf.constant(0,shape=shape)
    return tf.Variable(initial)
t7 = bias_variables([10])
print(t7)

#占位符
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])