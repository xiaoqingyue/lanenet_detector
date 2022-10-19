#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import cPickle as pickle
import matplotlib.pyplot as plt
import rospy


class LaneLine:
    """
    A lane line data class.

    """

    def __init__(self, params, yvals, curvered=0):
        if curvered == 0:
            self.params = params
            self.yvals = yvals
        else:
            self.params = params
            self.yvals = yvals
            self.curvered = curvered


class CurveredMiddleline:
    """
    Calculate the centerline and curvature of the lane line.

    """

    def __init__(self, lane_line, src_image):
 
        #self.lanelines=[]    
        self.curvatures = 0
        self.lane_lines = lane_line
        self.src_image = src_image
        self.perspective_path = rospy.get_param('~perspective_path')
        self.calibration_path = rospy.get_param('~calibration_path')

        self.data = pickle.load(open(self.perspective_path, 'rb'))

        #self.data = pickle.load(open("$(find lane_detector)/scripts/model/perspective/perspective_2.p", 'rb'))
        self.H = self.data['homography_matrix']
        self.H_inv = np.linalg.inv(self.data['homography_matrix'])
        self.y_pixels_per_meter = self.data['y_pixels_per_meter']
        self.x_pixels_per_meter = self.data['x_pixels_per_meter']        
        # self.img_center = src_image.shape[1] / 2
        self.calib = pickle.load(open(self.calibration_path, 'rb'))
        self.mtx = self.calib['mtx']
        self.rvec = self.calib['rvec']
        self.tvec = self.calib['tvec']

    def cameraToWorld(self, imgPoints):

        invK = np.asmatrix(self.mtx).I
        # rMat = np.zeros((3, 3), dtype=np.float64)
        # cv2.Rodrigues(r, rMat)
        # print('rMat=', rMat)
        rMat = self.rvec
        # 计算 invR * T
        invR = np.asmatrix(rMat).I  # 3*3
        # print('invR=', invR)
        transPlaneToCam = np.dot(invR, np.asmatrix(self.tvec))  # 3*3 dot 3*1 = 3*1
        # print('transPlaneToCam=', transPlaneToCam)
        #worldpt = []
        coords = np.zeros((3, 1), dtype=np.float64)
        
        pt = np.zeros((1199, 3), dtype=np.float64)

        for i, imgpt in enumerate(imgPoints):
            
            coords[0][0] = imgpt[0]
            coords[1][0] = imgpt[1]
            coords[2][0] = 1.0

            worldPtCam = np.dot(invK, coords)  # 3*3 dot 3*1 = 3*1
            # print('worldPtCam=', worldPtCam)
            # [x,y,1] * invR
            worldPtPlane = np.dot(invR, worldPtCam)  # 3*3 dot 3*1 = 3*1
            # print('worldPtPlane=', worldPtPlane)
            # zc
            scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
            # print("scale: ", scale)
            # zc * [x,y,1] * invR
            scale_worldPtPlane = np.multiply(scale, worldPtPlane)
            # print("scale_worldPtPlane: ", scale_worldPtPlane)
            # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
            worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
            # print("worldPtPlaneReproject: ", worldPtPlaneReproject)
           
            pt[i][0] = worldPtPlaneReproject[0][0]
            pt[i][1] = worldPtPlaneReproject[1][0]
            pt[i][2] = 0
            # worldpt.append(pt.T.tolist())
        worldpt=pt
            #worldpt.append(worldPtPlaneReproject)
        #print('imgpt:',pt)
        return worldpt


    def WorldToUTM(self, xyzPoints):
        # 实时车辆utm坐标
        map_utm_east = 524679.925327
        map_utm_north = 4061854.10789
        map_utm_yaw = -73.3227539062

        # 车辆的笛卡尔中的坐标 xyzPoints
        # 车辆的UTM坐标      utmPoints
        utmPoints = np.zeros((1199, 3), dtype=np.float64)
        yaw = 0
        utm_yaw = map_utm_yaw - yaw / math.pi * 180  # map_utm_yaw为轨迹原点处的航向, yaw为轨迹里某一点的笛卡尔坐标系的航向
        if utm_yaw > 180:
            utm_yaw = -(360.0 - utm_yaw)
        if utm_yaw < -180:
            utm_yaw += 360.0

        for i, imgpt in enumerate(xyzPoints):
            # 计算车辆在utm中的位姿
            if (map_utm_yaw > 0) | (map_utm_yaw == 0):
                map_yaw = 2 * math.pi - map_utm_yaw / 180 * math.pi
            if map_utm_yaw < 0:
                map_yaw = - map_utm_yaw / 180 * math.pi
            x_north = imgpt[0] * math.cos(map_yaw) - imgpt[1] * math.sin(map_yaw)
            y_east = -(imgpt[0] * math.sin(map_yaw) + imgpt[1] * math.cos(map_yaw))
            utm_north = map_utm_north + x_north
            utm_east = map_utm_east + y_east
            utmPoints[i][0] = utm_north
            utmPoints[i][1] = utm_east
            utmPoints[i][2] = utm_yaw
        # 输出值
        # utm_north, utm_east, utm_yaw
        return utmPoints

    def lines_identify(self):
        """
        We assume that the vehicle is located at the middle of the image.
        Lanes with nearby the middle of the images will be considered to belong the ego lane.
        The lanes consisted of ego lane will be marked by index 0.
        """
        y_range = [(line.yvals[-1], line.yvals[0]) for line in self.lane_lines]
        y_bottom_nearest_5p = [np.linspace(i[0] - 0.05 * (i[0] - i[1]), i[0], 10) for i in y_range]

        # Find the 5% points located at the bottom 找到位于底部的5％点
        x_bottom_5p_list = [np.poly1d(self.lane_lines[i].params)(y_bottom_nearest_5p[i]) for i in
                            range(len(self.lane_lines))]

        x_centers = [(np.sum(i) / i.size) for i in x_bottom_5p_list]
        # 将线进行左右排序
        s = np.array(x_centers)
        s_index = sorted(range(len(s)), key=lambda k: s[k])
        lanelines=[]
        for i in s_index:                             
            lanelines.append(self.lane_lines[i])
                                                                      
        img_center = self.src_image.shape[1] / 2  # 逆透视图的中
        distances = s - img_center
        left_lane_index = [s_index[i] for i in range(len(distances)) if distances[s_index[i]] <= 0]
        right_lane_index = [s_index[i] for i in range(len(distances)) if distances[s_index[i]] > 0]

        return left_lane_index, right_lane_index, lanelines

    def draw_pts(self, line, colour=np.array([0, 255, 0]).tolist()):
        """

        :param line:
        :param colour:
        """
        image = self.src_image
        x_vals = np.polyval(line.params, line.yvals)
        ipm_points_fit = np.vstack((x_vals, line.yvals)).transpose()
        pts = np.array([ipm_points_fit], dtype='float32')
        src_points = cv2.perspectiveTransform(pts, self.H_inv).squeeze(axis=0)
        pts = np.vstack((np.int_(src_points[:, 0]), np.int_(src_points[:, 1]))).transpose()
        cv2.polylines(image, [pts], False, colour, thickness=2)
        # plt.figure('middle')
        # plt.imshow(image[:, :, (2, 1, 0)])
        return image, pts

    # @property
    def middle_line(self):
        """
        :return:
        """
        global fit_x_top, fit_x_bottom, middle_curvatures
        left_lane_index, right_lane_index, lanelines = self.lines_identify()  # get the left lines and right lines
        lines_yvals = [line.yvals for line in self.lane_lines]
        coeffs = [line.params for line in self.lane_lines]
        # # 以下求中线 modified
        # max_position = ''
        # min_position = ''
        try:
            left_yy = lines_yvals[left_lane_index[-1]]
            right_yy = lines_yvals[right_lane_index[0]]
            max_com_yy = min(max(left_yy), max(right_yy))
            if max(left_yy) > max(right_yy):
                max_position = 'l'
            else:
                max_position = 'r'
            max_center_y = max(max(left_yy), max(right_yy))
            min_com_yy = max(min(left_yy), min(right_yy))
            if min(left_yy) < min(right_yy):
                min_position = 'l'
            else:
                min_position = 'r'
            min_center_y = min(min(left_yy), min(right_yy))
            # plot_y_center = np.linspace(min_center_y, max_center_y, max_center_y - min_center_y)
            p_left = np.poly1d(coeffs[left_lane_index[-1]])
            # fit_x_left = p_left(plot_y_center)
            p_right = np.poly1d(coeffs[right_lane_index[0]])
            # fit_x_right = p_right(plot_y_center)

            plot_y_com_center = np.linspace(min_com_yy, max_com_yy, max_com_yy - min_com_yy)
            fit_x_com_left = p_left(plot_y_com_center)
            fit_x_com_right = p_right(plot_y_com_center)
            fit_x_com_center = (fit_x_com_left + fit_x_com_right) / 2  # 中线横坐标
            if min_com_yy + 2 > min_center_y:  # 前端纵坐标点
                plot_y_center_top = np.linspace(min_center_y, min_com_yy, min_com_yy - min_center_y)
                if min_position == 'l':
                    fit_x_top = p_left(plot_y_center_top) + (p_left(min_com_yy) + p_right(min_com_yy)) / 2 - p_left(
                        min_com_yy)
                elif min_position == 'r':
                    fit_x_top = p_right(plot_y_center_top) + (p_left(min_com_yy) + p_right(min_com_yy)) / 2 - p_right(
                        min_com_yy)
            else:
                plot_y_center_top = []
                fit_x_top = []
            if max_com_yy + 2 < max_center_y:  # 后端横坐标
                plot_y_center_bottom = np.linspace(max_com_yy, max_center_y, max_center_y - max_com_yy)
                if max_position == 'l':
                    fit_x_bottom = p_left(plot_y_center_bottom) + (
                            p_left(max_com_yy) + p_right(max_com_yy)) / 2 - p_left(max_com_yy)
                elif max_position == 'r':
                    fit_x_bottom = p_right(plot_y_center_bottom) + (
                            p_left(max_com_yy) + p_right(max_com_yy)) / 2 - p_right(max_com_yy)
            else:
                plot_y_center_bottom = []
                fit_x_bottom = []
            # fit_x_center = np.hstack((fit_x_top, fit_x_com_center, fit_x_bottom))
            fit_x_center = np.hstack((fit_x_com_center, fit_x_bottom))  # no top part
            # plot_y_center = np.hstack((plot_y_center_top, plot_y_com_center, plot_y_center_bottom))
            plot_y_center = np.hstack((plot_y_com_center, plot_y_center_bottom))  # no top part
            if len(plot_y_center) == 0:
                plot_y_center = plot_y_center_top
                fit_x_center = fit_x_top
        except:
            return []
        # 俯视图拟合中线
        try:
            fit_param_center = np.polyfit(plot_y_center, fit_x_center, 2)
        except TypeError:
            return []
        p_center = np.poly1d(fit_param_center)
        fit_x_center = p_center(plot_y_center)
        # Fit in real world coordinates
        # y_pixels_per_meter = self.y_pixels_per_meter
        y_pixels_per_meter = self.x_pixels_per_meter
        plot_y_center_real = (self.src_image.shape[0] - plot_y_center) / y_pixels_per_meter
        fit_x_center_real = (fit_x_center - self.src_image.shape[1] / 2) / y_pixels_per_meter
        # fit_param_center_real = np.polyfit(plot_y_center_real, fit_x_center_real, 2)
        middle_line = LaneLine(fit_param_center, plot_y_center)
        middleline_image, middleline_point = self.draw_pts(middle_line, np.array([0, 255, 0]).tolist())

        curvatures = self.abnormal_curve(middle_line.yvals[0], middle_line.yvals[-1], middle_line.params)
        if curvatures[0]:
            middle_curvatures=curvatures[1]

        middleline_world = self.cameraToWorld(middleline_point)
        
        middleline_utm = self.WorldToUTM(middleline_world)

        result = {
            'lane_lines': lanelines,
            'middle_line': middle_line,
            'middle_curvatures': middle_curvatures,
            'result_image': middleline_image,
            'middleline_world':middleline_world,
            'middleline_utm':middleline_utm
        }        
        return result

    def polynominal_2nd_derivate(self, coeff, x):  # 2nd derivate of polynominal of 3 or 2 order
        if len(coeff) == 4:
            return 6 * coeff[0] * x + 2 * coeff[1]
        elif len(coeff) == 3:
            return 2 * coeff[0]

    def polynominal_derivate(self, coeff, x):  # derivate of polynominal of 3 or 2 order
        if len(coeff) == 4:
            return 3 * coeff[0] * x ** 2 + 2 * coeff[1] * x + coeff[2]
        elif len(coeff) == 3:
            return 2 * coeff[0] * x + coeff[1]

    def get_curvature(self, coeff, x):
        """
        Calculates the curvature using r = [1 + (dy/dx)^2]^3/2 / |d^2y/dx^2|
        adds curvature to self.curvature,
        :return: curvature appended as self.curvature
        """
        return np.absolute(self.polynominal_2nd_derivate(coeff, x)) / (
                (1 + self.polynominal_derivate(coeff, x) ** 2) ** 1.5)

    def abnormal_curve(self, start_y, end_y, coeff):
        ''' Recognize the abnormal curve. 识别异常曲线'''
        y_num = 10
        y_pts = np.linspace(start_y, end_y, y_num)

        _2nd_derivate = [self.polynominal_2nd_derivate(coeff, x) for x in y_pts]  # 二阶导数
        _2nd_derivate = _2nd_derivate[1:-1]
        curvatures = [self.get_curvature(coeff, i) for i in y_pts]  # 曲率半径
        threshold = 15
        num = 3
        curvatures_avg_top = sum([curvatures[i] for i in range(num)]) / num
        curvatures_avg_bottom = sum([curvatures[-i - 1] for i in range(num)]) / num

        if _2nd_derivate[0] * _2nd_derivate[-1] < 0:  # detect the s curvature line. 检测s曲率线
            if np.absolute(curvatures_avg_top + curvatures_avg_bottom) > np.absolute(
                    threshold * 1.5 * curvatures_avg_bottom):  # 不取上端点
                return False,

        if np.absolute(curvatures_avg_top - curvatures_avg_bottom) > np.absolute(
                threshold * curvatures_avg_bottom):  # 不取上端点
            return False,
        return True, sum(curvatures) / y_num

    def curvatures_line(self):
        """ Delete invalid lines. """
        lines = []
        avg_curvatures = []

        for line in self.lane_lines:

            cur = self.abnormal_curve(line.yvals[0], line.yvals[-1], line.params)
            if cur[0]:                
                avg_curvatures.append(cur[1])
                lines.append(LaneLine(line.params, line.yvals, cur[1]))

        if len(avg_curvatures) < 1:
            self.lane_lines = []
            return []
        if len(
                avg_curvatures) > 2:  # If there are more than 2 lines, filter the lines according to the curvatures consistency
            avg_value = sum(avg_curvatures) / len(avg_curvatures)
            dis = sorted(np.absolute(avg_curvatures - avg_value))
            dis_threshold = 5
            filted_lines = []
            for i in range(len(avg_curvatures)):
                if avg_curvatures[i] - avg_value > dis_threshold * dis[-2]:
                    continue
                filted_lines.append(lines[i])
            self.lane_lines = filted_lines

        return avg_curvatures


if __name__ == '__main__':
    pass
