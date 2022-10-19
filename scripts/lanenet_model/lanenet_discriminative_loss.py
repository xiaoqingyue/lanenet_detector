#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午3:48
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_discriminative_loss.py
# @IDE: PyCharm Community Edition
"""
Discriminative Loss for instance segmentation
"""
import tensorflow as tf


def discriminative_loss_single(
        prediction,
        correct_label,
        feature_dim,
        label_shape,
        delta_v,
        delta_d,
        param_var,
        param_dist,
        param_reg):
    """
    discriminative loss
    :param prediction: inference of network  网络输出
    :param correct_label: instance label     实例标签
    :param feature_dim: feature dimension of prediction   4
    :param label_shape: shape of label
    :param delta_v: cut off variance distance  切断方差 0.5
    :param delta_d: cut off cluster distance    聚类距离 3.0
    :param param_var: weight for intra cluster variance   集群内方差的权重 1.0
    :param param_dist: weight for inter cluster distances   集群间距离的权重 1.0
    :param param_reg: weight regularization                 正则化 0.001
    """
    correct_label = tf.reshape(   # 像素对齐为一行
        correct_label, [label_shape[1] * label_shape[0]]
    )
    reshaped_pred = tf.reshape(
        prediction, [label_shape[1] * label_shape[0], feature_dim]
    )

    # calculate instance nums  # 统计实例个数
    # unique_labels统计出correct_label中一共有几种数值，unique_id为correct_label中的每个数值是属于unique_labels中第几类
    # counts统计unique_labels中每个数值在correct_label中出现了几次
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)  # 元素个数  表示有几条车道线

    # calculate instance pixel embedding mean vec
    # 计算pixel embedding均值向量
    # segmented_sum是把reshaped_pred中对于GT里每个部分位置上的像素点相加
    # 比如unique_id[0, 0, 1, 1, 0],reshaped_pred[1, 2, 3, 4, 5]，最后等于[1+2+5,3+4],channel层不相加
    segmented_sum = tf.unsorted_segment_sum(
        reshaped_pred, unique_id, num_instances)   # reshaped_pred 预测标签 需要计算的数据
    # [num_instances, feature_dim]
    # 除以每个类别的像素在gt中出现的次数，是每个类别像素的均值 (?, feature_dim)

    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))  # 聚类均值# [num_instances, feature_dim]
    # 然后再还原为原图的形式，现在mu_expand中的数据是与correct_label的分布一致，但是数值不一样
    mu_expand = tf.gather(mu, unique_id)   # 将张量对应索引的向量提取出来 # 特征均值矩阵

    # 计算公式的loss(var)
    # 对channel维度求范数-[131072，]
    distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)  # 计算向量、矩阵和张量的范数.# gb& ||mu_c - x_i||
    distance = tf.subtract(distance, delta_v)   # bg& ||mu_c - x_i|| - delta_v
    # 小于0的设置为0，大于distance的设置为distance
    distance = tf.clip_by_value(distance, 0., distance)  # 截取distance使之在min和max之间  # bg& (||mu_c - x_i|| - delta_v)_+
    distance = tf.square(distance)  # bg& (||mu_c - x_i|| - delta_v)_+^2
    # 方差项
    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)  # bg& (||mu_c - x_i|| - delta_v)_+^2
    l_var = tf.div(l_var, counts)  # (||mu_c - x_i|| - delta_v)_+^2 / N_c
    l_var = tf.reduce_sum(l_var)   # sumC{sum[(||mu_c - x_i|| - delta_v)_+^2 / N_c]}
    # sumC{sum[(||mu_c - x_i|| - delta_v)_+^2 / N_c]} / C
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    # 距离项 # 计算公式的loss(dist)
    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])  # 0 轴方向上复制 num_instances 次
    mu_band_rep = tf.tile(mu, [1, num_instances])         # 1 轴方向上复制 num_instances 次
    mu_band_rep = tf.reshape(
        mu_band_rep,
        (num_instances *
         num_instances,
         feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)    # mu_ca - mu_cb

    # 去除掩模上的零点 ca != cb
    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)  # 将输入的数组挑出想要的数据输出

    mu_norm = tf.norm(mu_diff_bool, axis=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)   # (2*delta_d - ||mu_ca - mu_cb||)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)  # (2*delta_d - ||mu_ca - mu_cb||)_+
    mu_norm = tf.square(mu_norm)  # (2*delta_d - ||mu_ca - mu_cb||)_+^2

    l_dist = tf.reduce_mean(mu_norm)  # sum[2*delta_d - ||mu_ca - mu_cb||)_+^2] / C

    # 计算原始Discriminative Loss论文中提到的正则项损失
    l_reg = tf.reduce_mean(tf.norm(mu, axis=1))  # mean[sqrt(sum(mu_c^2))]

    # 合并损失按照原始Discriminative Loss论文中提到的参数合并
    param_scale = 1.
    l_var *= param_var
    l_dist *= param_dist
    l_reg *= param_reg

    loss = param_scale * (l_var + l_dist + l_reg)  # loss = l_var + l_dist + 0.001*mean[sqrt(sum(mu_c^2))]

    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    """

    :return: discriminative loss and its three components
    """

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(
            prediction[i], correct_label[i], feature_dim, image_shape, delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing  TensorArray是一种支持动态写入的数据结构
    output_ta_loss = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(
        cond, body, [
            correct_label, prediction, output_ta_loss, output_ta_var, output_ta_dist, output_ta_reg, 0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)  # 计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg