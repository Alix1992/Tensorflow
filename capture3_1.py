# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:36:45 2019

@author: Administrator
"""

import tensorflow as tf
a =tf.constant([1.0,2.0],name="a")
b =tf.constant([1.0,2.0],name="b")
result = a+b
sess = tf.Session()
print(sess.run(result))
print(tf.get_default_graph())

g1= tf.Graph()
with g1.as_default():
        v = tf.get_variable("v",initializer=tf.zeros_initializer()(shape=[1]))
        ''' 书本上tensorflow 0.9.0是initializer=tf.zeros_initializer(shape=[1]) 安装tensorflow1.2initializer=tf.zeros_initializer()(shape=[1])'''
        
g2= tf.Graph()
with g2.as_default():
        v = tf.get_variable("v",initializer=tf.ones_initializer()(shape=[1]))
       
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

''' 指定设备 将Graph用特定设备运行'''
g= tf.Graph()
with g.device('/gpu:0'):
    result = a+b
sess = tf.Session()
print(sess.run(result))     


'''张量'''
a = tf.constant([1.0,2.0],name="a",dtype="float32")
b = tf.constant([2.0,3.0],name="b",dtype="float32")
result = tf.add(a,b,name="add")
sess = tf.Session()
print(result)
print(sess.run(result))




   