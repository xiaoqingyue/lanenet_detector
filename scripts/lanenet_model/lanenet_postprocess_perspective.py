#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math
import matplotlib.pyplot as plt
import rospy
import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config import global_config

#import pickle
import cPickle as pickle

from lanenet_model.calculate_curvered_middleline import LaneLine
from lanenet_model.calculate_curvered_middleline import CurveredMiddleline

CFG = global_config.cfg


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result  形态学过程填补了二元分割结果中的空洞
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components 连接组件分析以删除较小的组件
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)  # 8邻域连通区域


class _LaneFeat(object):
    """

    """

    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    @staticmethod
    def _embedding_feats_dbscan_cluster(embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=CFG.POSTPROCESS.DBSCAN_EPS, min_samples=CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(
                embedding_image_feats)  # StandardScaler作用：去均值和方差归一化,且是针对每一个特征维度来做的，而不是针对样本。
            db.fit(features)
        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)  # 去除重复的元素

        num_clusters = len(unique_labels)  # 类别
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result 根据二进制段结果获取车道嵌入特征
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]

        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords 获得binnary车道线坐标点和instance车道线点
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,  # lane_coordinates 9902*2
            instance_seg_ret=instance_seg_result  # lane_embedding_feats 9902*4
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']  # [-1 0 1 2 3] -1表示噪点
        coord = get_lane_embedding_feats_result['lane_coordinates']  # 获得binnary车道线坐标点

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):  # #数组转列表
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))  # tuple()函数将列表转换为元组
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """

    def __init__(self):
        """

        :param ipm_remap_file_path: ipm generate file path
        """

        self._cluster = _LaneNetCluster()
        self.perspective_path = rospy.get_param('~perspective_path')

        self.data = pickle.load(open(self.perspective_path, 'rb'))
        self.H = self.data['homography_matrix']
        self.H_inv = np.linalg.inv(self.data['homography_matrix'])
        self.y_boundary = self.data['vanishing_point'][1] + self.data['margin_x_y'][1]


        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=150, source_image=None):
        """
        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        height, width = source_image.shape[0], source_image.shape[1]
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area  形态学处理
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=6)
        plt.figure('morphological_ret1')
        plt.imshow(morphological_ret)

        # 获得连通区域
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)
        # 去除很小的连通域
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )
        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        if mask_image is None or len(lane_coords) < 2:
            return {
                'lane_lines': None,
                'source_image': source_image,
            }

        # 可视化mask图
        src_mask_scaled = np.zeros((256, 512, 3))  # 256*512大小的原图mask
        src_mask_thinning = np.zeros((height, width, 3))  # 原始图像空间  车道线经过细化之后的mask
        ipm_mask = np.zeros((height, width, 3))  # 鸟瞰图下的mask

        lane_lines = []
        for lane_index, coords in enumerate(lane_coords):
            if lane_coords is None:
                continue

            cv2.polylines(src_mask_scaled, [coords], False, self._color_map[lane_index].tolist(), thickness=1)
            plt.figure('[src_mask_scaled]')
            plt.imshow(src_mask_scaled / 255)

            x_original, y_original = coords[:, 0] * width / 512, coords[:, 1] * height / 256

            # 原图下车道线细化
            coords_sampled = []
            for y in np.unique(y_original):  # 去除数组中重复的数据,并且排列之后输出
                if y < self.y_boundary:
                    continue
                idx = np.where(y_original == y)
                x = np.sum(x_original[idx[0]]) / len(x_original[idx[0]])
                cv2.circle(src_mask_thinning, (np.int(x), np.int(y)), 3, self._color_map[lane_index].tolist(), -1)
                coords_sampled.append((x, y))

            # plt.figure('[src_mask_thinning]')
            # plt.imshow(src_mask_thinning / 255)

            coords_sampled = np.array(coords_sampled)

            pts = np.array([coords_sampled], dtype='float32')

            if pts.size == 0:
                continue

            ipm_points = cv2.perspectiveTransform(pts, self.H).squeeze(axis=0)  # 二维矢量透视变换

            # 鸟瞰图下沿y方向非均匀采样
            # step 1 剔除远端部分点
            delete_points_num = int(0.02 * len(ipm_points))
            ipm_points = np.delete(ipm_points, np.arange(0, delete_points_num, 1), axis=0)
            ipm_points = np.delete(ipm_points, np.arange(len(ipm_points) - delete_points_num, len(ipm_points), 1),
                                   axis=0)
            x_ipm, y_ipm = ipm_points[:, 0], ipm_points[:, 1]

            # step 2 远端稀疏采样，近端稠密采样
            far_ratio, far_interval, near_interval = 0.3, 2, 1

            far_pixel_num = int(far_ratio * (y_ipm[-1] - y_ipm[0]))
            near_pixel_num = int((1 - far_ratio) * (y_ipm[-1] - y_ipm[0]))

            far_pixel_interp_num = int(far_pixel_num / far_interval)  # 远端像素个数
            near_pixel_interp_num = int(near_pixel_num / near_interval)  # 近端像素个数

            yvals_far = np.linspace(y_ipm[0], y_ipm[0] + far_pixel_num, far_pixel_interp_num)
            yvals_near = np.linspace(y_ipm[-1] - near_pixel_num, y_ipm[-1], near_pixel_interp_num)
            yvals = np.concatenate((yvals_far, yvals_near), axis=0)  # 多个数组的拼接
            # yvals = np.linspace(np.min(y_ipm), np.max(y_ipm), int((np.max(y_ipm) - np.min(y_ipm)) / 5))

            if len(yvals) == 0:
                continue
            # 插值
            x_interp = np.interp(yvals, y_ipm, x_ipm)
            # 拟合
            # yvals = yvals - height

            p = np.polyfit(yvals, x_interp, 2)

            plot_y = np.linspace(0, height - 1, height)

            x_fit = np.polyval(p, plot_y)

            lane_lines.append(LaneLine(p, plot_y))

            # yvals = yvals + height

            # 在鸟瞰图上绘制车道线
            lane_fit = np.vstack((np.int_(x_fit), np.int_(plot_y))).transpose()
            lane_fit = lane_fit.reshape((-1, 1, 2))
            cv2.polylines(ipm_mask, [lane_fit], False, self._color_map[lane_index].tolist(), thickness=6)
            # plt.figure('[ipm_mask]')
            # plt.imshow(ipm_mask / 255)

            # 原始检测结果从鸟瞰图到透视图下变换
            ipm_points_fit = np.vstack((x_fit, plot_y)).transpose()

            pts = np.array([ipm_points_fit], dtype='float32')

            src_points = cv2.perspectiveTransform(pts, self.H_inv).squeeze(axis=0)  # 对二维数组进行反逆透视变换

            pts = np.vstack((np.int_(src_points[:, 0]), np.int_(src_points[:, 1]))).transpose()
            cv2.polylines(source_image, [pts], False, self._color_map[2].tolist(), thickness=6)
            # plt.figure('[src_image]')
            # plt.imshow(src_image[:, :, (2, 1, 0)])
            # plt.show()

        ret = {
            'lane_lines': lane_lines,
            'source_image': source_image,
        }

        return ret

    def postprocess_curv_middle(self, lane_lines, src_image):
        
        """
        :param lane_lines:
        :param src_image:
        :return:
        """
        lane_post_process = CurveredMiddleline(lane_lines, src_image)
        lane_post_process.curvatures_line()
        result = lane_post_process.middle_line()
      
        # result = {
        #     'lane_lines': lane_post_process.lanelines,
        #     'middle_line': middle_line,
        #     'result_image': result_image,
        # }

        return result
