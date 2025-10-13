# -*- coding: utf-8 -*-
"""
煤堆分割系统配置文件
包含所有可配置的参数值
"""

import logging
import os
import struct


HEADER_FORMAT = "<4s HHHHIIdd II QQ II 3d 12d 12d 15d I 6d"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
POINT_FORMAT = "<iiiBB"
POINT_SIZE = struct.calcsize(POINT_FORMAT)

class GrabPointCalculationConfig_170:
    """
    抓取点计算配置
    """
    # line_width = 4      # 线宽
    # floor_height = 2  # 层高

    # safe_distance_x_negative_init=5  #大车左侧安全距离
    # safe_distance_x_positive_init=5  #大车右侧安全距离
    # safe_distance_y_ocean_init=5  #海侧安全距离
    # safe_distance_y_land_init=5  #陆侧安全距离

    # hatch_depth=16  # 型深
    # retain_height=0.5 #保留高度
    # line_gap=0.5 #线和线之间最小的间隔

    # expansion_x_front=0.4  #前四层大车方向（x）外扩系数.
    # expansion_y_front=0.5  #前四层小车方向（y）外扩系数.
    
    # expansion_x_back=0.9  #后四层大车方向（x）外扩系数.
    # expansion_y_back=0.9  #后四层小车方向（y）外扩系数.
    # plane_threshold = 0.3   #平面阈值
    # height_diff=3.5  #高度差

    # Sign =False      #True使用线性函数
    # k=2
    # b=-1
    # plane_distance=0.5 #平面的情况抓取点移动的距离
    # bevel_distance=1.5  #斜面的情况抓取点移动的距离 #-1.8



    # block_width=1  #每个分块的宽度
    # block_length=1  #每个分块的长度


    # WebSocket配置
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 27052
    SERVER_PATH = "/point-cloud-170"
    USER_ID = "coal-pile-detector-170"

    # 广播服务器配置
    BROADCAST_HOST = "192.168.1.222"  # 根据实际网络配置调整
    BROADCAST_PORT = 8765  # 使用不同的端口避免冲突


    VIS_ORIGINAL_3D = False  # 可视化原始点云
    VISUALIZE_COARSE_FILTRATION = False   # 可视化粗过滤结果
    VISUALIZE_RECTANGLE = False  # 可视化矩形检测过程
    VISUALIZE_FINAL_RESULT = False      # 可视化最终结果
    visualize_clustered_pcd = False  # 可视化聚类后的结果
    visualize_coal_points = False  # 可视化煤堆点

    # 安装参数
    translation = [1.58, 24.324, 31.4]
        
    # 旋转角度 [roll, pitch, yaw] 
    rotation_angles = [-6.05, 0.15, 0]

    # 粗过滤coarse_filtration_point_cloud函数的参数
    x_initial_threshold=10.0 
    x_stat_percentage=0.3
    x_filter_offset=3
    intensity_threshold=10.0 #强度
    x_upper_bound_adjustment=0

    # 矩形检测find_rectangle_by_histogram_method函数的参数
    pixel_size=0.1 #栅格化大小
    kernel_size=18  #核大小
    white_threshold=0.4  #白色像素的比例



    eps_coal = 1.3
    min_samples_coal = 20
    x_changes=5
    yz_shrink_amount=0.6 #矩形框向内收缩的比例

    #真实坐标系下的参数
    z_changes=5
    xy_shrink_amount=0.6 #矩形框向内收缩的比例







    
    limited_layers=6  #开启海陆侧甩斗限制的层数
    limited_height=4.5  #限制的层数高度
    limited_change_height=1.4  #海陆侧甩斗向上加一定距离

    #小车方向甩斗的限制层数
    limited_layers_y_dump_truck=2.5

    #大车方向甩斗的限制层数
    limited_layers_x_dump_truck=2


    x_dump_truck=2             #距离大车方向的安全边界的距离阈值，判断是否在大车方向甩斗
    y_dump_truck=3             #距离小车方向的安全边界的距离阈值，判断是否在小车方向甩斗


    y_grab_expansion=3.1    #海陆侧向外多延申的距离


    #分割海陆侧的参数
    land_to_centerline=2  #距离中心线的距离阈值，判断是否在陆侧
    ocean_to_centerline=2  #距离中心线的距离阈值，判断是否在海侧





class GrabPointCalculationConfig_160:
    """
    抓取点计算配置
    """
    # line_width = 4      # 线宽
    # floor_height = 2  # 层高

    # safe_distance_x_negative_init=5  #大车左侧安全距离
    # safe_distance_x_positive_init=5  #大车右侧安全距离
    # safe_distance_y_ocean_init=5  #海侧安全距离
    # safe_distance_y_land_init=5  #陆侧安全距离

    # hatch_depth=16  # 型深
    # retain_height=0.5 #保留高度
    # line_gap=0.5 #线和线之间最小的间隔

    # expansion_x_front=0.4  #前四层大车方向（x）外扩系数.
    # expansion_y_front=0.5  #前四层小车方向（y）外扩系数.
    
    # expansion_x_back=0.9  #后四层大车方向（x）外扩系数.
    # expansion_y_back=0.9  #后四层小车方向（y）外扩系数.
    # plane_threshold = 0.3   #平面阈值
    # height_diff=3.5  #高度差

    # Sign =False      #True使用线性函数
    # k=2
    # b=-1
    # plane_distance=0.5 #平面的情况抓取点移动的距离
    # bevel_distance=1.5  #斜面的情况抓取点移动的距离 #-1.8



    # block_width=1  #每个分块的宽度
    # block_length=1  #每个分块的长度


    # WebSocket配置
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 27052
    SERVER_PATH = "/point-cloud-160"
    USER_ID = "coal-pile-detector-160"

    # 广播服务器配置
    BROADCAST_HOST = "192.168.1.222"  # 根据实际网络配置调整
    BROADCAST_PORT = 8766  # 使用不同的端口避免冲突


    VIS_ORIGINAL_3D = False  # 可视化原始点云
    VISUALIZE_COARSE_FILTRATION = False   # 可视化粗过滤结果
    VISUALIZE_RECTANGLE = False  # 可视化矩形检测过程
    VISUALIZE_FINAL_RESULT = False      # 可视化最终结果
    visualize_clustered_pcd = False  # 可视化聚类后的结果
    visualize_coal_points = False  # 可视化煤堆点

    # 安装参数
    translation = [1.58, 24.324, 31.4]
        
    # 旋转角度 [roll, pitch, yaw] 
    rotation_angles = [-6.05, 0.15, 0]

    # 粗过滤coarse_filtration_point_cloud函数的参数
    x_initial_threshold=10.0 
    x_stat_percentage=0.3
    x_filter_offset=3
    intensity_threshold=10.0 #强度
    x_upper_bound_adjustment=0

    # 矩形检测find_rectangle_by_histogram_method函数的参数
    pixel_size=0.1 #栅格化大小
    kernel_size=18  #核大小
    white_threshold=0.4  #白色像素的比例



    eps_coal = 1.3
    min_samples_coal = 20
    x_changes=5
    yz_shrink_amount=0.6 #矩形框向内收缩的比例


    #真实坐标系下的参数
    z_changes=5
    xy_shrink_amount=0.6 #矩形框向内收缩的比例




    
    limited_layers=6  #开启海陆侧甩斗限制的层数
    limited_height=4.5  #限制的层数高度
    limited_change_height=1.4  #海陆侧甩斗向上加一定距离

    #小车方向甩斗的限制层数
    limited_layers_y_dump_truck=2.5

    #大车方向甩斗的限制层数
    limited_layers_x_dump_truck=2


    x_dump_truck=2             #距离大车方向的安全边界的距离阈值，判断是否在大车方向甩斗
    y_dump_truck=3             #距离小车方向的安全边界的距离阈值，判断是否在小车方向甩斗


    y_grab_expansion=3.1    #海陆侧向外多延申的距离