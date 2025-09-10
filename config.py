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
    safe_distance_init=5  #安全距离
    line_gap=0 #线和线之间的间隔

    expansion_x_front=0.4  #前四层大车方向（x）外扩系数.
    expansion_y_front=0.5  #前四层小车方向（y）外扩系数.
    
    expansion_x_back=0.9  #后四层大车方向（x）外扩系数.
    expansion_y_back=0.9  #后四层小车方向（y）外扩系数.
    plane_threshold = 0.02   #平面阈值
    plane_distance=-0.5  #平面的情况抓取点移动的距离
    bevel_distance=1  #斜面的情况抓取点移动的距离

    block_width=1  #每个分块的宽度
    block_length=1  #每个分块的长度
    
    plane_ratio=0.5  #平面占比50%
    bevel_ratio=0.3  # 斜面占比30%
