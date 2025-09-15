# -*- coding: utf-8 -*-
"""
煤堆分割系统配置文件
包含所有可配置的参数值
"""

import logging
import os

class GrabPointCalculationConfig:
    """
    抓取点计算配置
    """
    line_width = 4      # 线宽
    floor_height = 2  # 层高

    safe_distance_x_negative_init=5  #大车左侧安全距离
    safe_distance_x_positive_init=5  #大车右侧安全距离
    safe_distance_y_ocean_init=5  #海侧安全距离
    safe_distance_y_land_init=5  #陆侧安全距离

    hatch_depth=16  # 型深
    retain_height=0.5 #保留高度
    line_gap=0.5 #线和线之间最小的间隔

    expansion_x_front=0.4  #前四层大车方向（x）外扩系数.
    expansion_y_front=0.5  #前四层小车方向（y）外扩系数.
    
    expansion_x_back=0.9  #后四层大车方向（x）外扩系数.
    expansion_y_back=0.9  #后四层小车方向（y）外扩系数.
    plane_threshold = 0.3   #平面阈值
    height_diff=3.5  #高度差
    plane_distance=-1  #平面的情况抓取点移动的距离
    bevel_distance=1  #斜面的情况抓取点移动的距离



    block_width=1  #每个分块的宽度
    block_length=1  #每个分块的长度
    
