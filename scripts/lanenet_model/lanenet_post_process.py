#!/usr/bin/env python3
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

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config import global_config

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

    # kernel1 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
    #
    # closing = cv2.dilate(closing, kernel1)
    # #
    # kernel1 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(1, 1))
    # # #
    # # closing = cv2.erode(dilating, kernel1)
    # closing = cv2.dilate( closing , kernel)
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

        num_clusters = len(unique_labels)
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
        ipm_remap_file_path = '/home/zhy/Documents/Perception/Lanelinedetection/DNN/LaneNet/lanenet-lane-detection' \
                              '/data/jinan_ipm_remap.yml'
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cluster = _LaneNetCluster()

        remap_file_load_ret = self._load_remap_matrix(ipm_remap_file_path)

        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        pm_remap_file_path = '/home/zhy/Documents/Perception/Lanelinedetection/DNN/LaneNet/lanenet-lane' \
                             '-detection/data/jinan_pm_remap.yml'
        assert ops.exists(pm_remap_file_path), '{:s} not exist'.format(pm_remap_file_path)
        remap_file_load_pm = self._load_remap_matrix(pm_remap_file_path)

        self._remap_to_pm_x = remap_file_load_pm['remap_to_ipm_x']
        self._remap_to_pm_y = remap_file_load_pm['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def _load_remap_matrix(self, file_path):
        """

        :return:
        """
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)  # 加载YMl文件数据

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=150, source_image=None,
                    data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        global plot_y
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area  形态学处理
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=6)


        # 获得连通区域
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)
        # 去除很小的连通域
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # 获取感兴趣区域
        mask = np.zeros(shape=(256, 512), dtype=np.uint8)
        ptx1 = np.array([[0, 256], [250, 110], [270, 110], [145, 256]])
        # ptx = np.array([[300, 250], [260, 135], [295, 135], [300, 140], [320, 180], [460, 250]])
        ptx = np.array([[300, 256], [260, 110], [280, 110], [460, 256]])
        cv2.fillConvexPoly(mask, ptx, (255, 0, 0))
        cv2.fillConvexPoly(mask, ptx1, (255, 0, 0))
        morphological_ret = cv2.bitwise_and(morphological_ret, mask)

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )
    
        if mask_image is None or len(lane_coords) < 2 or len(lane_coords) > 2:
            return {
                'lane_center': None,
                'fit_params': None,
                'source_image': source_image,
            }

        # lane line fit
        fit_params = []
        lane_fit_x = []  # lane pts every single lane 每条车道的点数
        remap_mask = np.zeros(shape=(1200, 1920, 3), dtype=np.uint8)
        for lane_index, coords in enumerate(lane_coords):  # lane_coords 每条车道线的坐标点
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(1200, 1920), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 1200 / 256), np.int_(coords[:, 0] * 1920 / 512)))] = 255
            elif data_source == 'beec_ccd':
                tmp_mask = np.zeros(shape=(1350, 2448), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 1350 / 256), np.int_(coords[:, 0] * 2448 / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
        
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )  # 一个坐标到另一个坐标的映射
         

            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])  # 分别取对应位置的值组成非零元素的坐标

            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)  # 多项式拟合  返回二次多项式系数
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            # ipm_image_start = nonzero_y.min()
            plot_y = np.linspace(0, ipm_image_height - 1, ipm_image_height)  # 创建等差数列
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]
            j = 0
            for i in range(len(fit_x) - 1):
                if fit_x[i] >= 0:
                    if fit_x[i] > fit_x[i + 1]:
                        j += 1
                else:
                    return {
                        'mask_image': None,
                        'fit_params': None,
                        'source_image': source_image,
                    }
            if j > 200:
                return {
                    'lane_center': None,
                    'fit_params': None,
                    'source_image': source_image,
                }
            # 在鸟瞰图上绘制车道线
            lane_ptc = np.vstack((np.int_(fit_x), np.int_(plot_y))).transpose()
            lane_ptc = lane_ptc.reshape((-1, 1, 2))
            cv2.polylines(remap_mask, [lane_ptc], False, (0, 255, 0), 4)

            lane_fit_x.append(fit_x)  # 保存拟合后车道线

        # 求车道线中线
        center_remap_mask = np.zeros(shape=(1200, 1920), dtype=np.uint8)
        if lane_fit_x[1][0] > lane_fit_x[0][0]:
            center_x = (abs(lane_fit_x[1] - lane_fit_x[0]) / 2) + lane_fit_x[0]
        else:
            center_x = (abs(lane_fit_x[1] - lane_fit_x[0]) / 2) + lane_fit_x[1]

        center_ptc = np.vstack((np.int_(center_x), np.int_(plot_y))).transpose()

        center_remap_mask[tuple((np.int_(center_ptc[:, 1]), np.int_(center_ptc[:, 0])))] = 255

        center_ptc = center_ptc.reshape((-1, 1, 2))
        cv2.polylines(remap_mask, [center_ptc], False, (0, 0, 255), 4)

        tmp_pm_mask = cv2.remap(
            remap_mask,
            self._remap_to_pm_x,
            self._remap_to_pm_y,
            interpolation=cv2.INTER_NEAREST
        )  # 一个坐标到另一个坐标的映射
        source_image = cv2.addWeighted(source_image, 1, tmp_pm_mask, 0.3, 0)

        center_pm_mask = cv2.remap(
            center_remap_mask,
            self._remap_to_pm_x,
            self._remap_to_pm_y,
            interpolation=cv2.INTER_NEAREST
        )

        center_nonzeroy = np.array(center_pm_mask.nonzero()[0])
        center_nonzerox = np.array(center_pm_mask.nonzero()[1])
        center_image_start = center_nonzeroy.min()

        center_fits_param = np.polyfit(center_nonzeroy, center_nonzerox, 3)  # 多项式拟合  返回二次多项式系数

        [center_image_height, center_image_width] = center_pm_mask.shape

        ploty = np.linspace(center_image_start, center_image_height - 1,
                            center_image_height - center_image_start)  # 创建等差数列
        # fitx = fits_param[0] * ploty ** 2 + fits_param[1] * ploty + fits_param[2]
        center_fitx = center_fits_param[0] * ploty ** 3 + center_fits_param[1] * ploty ** 2 + center_fits_param[
            2] * ploty + center_fits_param[3]

        centerptc = np.vstack((np.int_(center_fitx), np.int_(ploty))).transpose()
        centerptc = centerptc.reshape((-1, 1, 2))
        cv2.polylines(source_image, [centerptc], False, (0, 0, 255), 4)

        lane_center = np.vstack((np.int_(center_fitx).transpose(), np.int_(ploty).transpose())).transpose()

        ret = {
            'lane_center': lane_center,
            'fit_params': fit_params,
            'source_image': source_image,
        }

        return ret
