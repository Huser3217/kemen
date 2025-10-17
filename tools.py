
# 导入所需的库
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING) 
import laspy 
import numpy as np 
import cv2 
import open3d as o3d
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from sklearn.cluster import DBSCAN 
from scipy.spatial import ConvexHull 
from sklearn.neighbors import KDTree 
from scipy.spatial.transform import Rotation as R
import hdbscan
import time
import logging
import os
# WebSocket相关导入
import asyncio
import websockets
import json
import base64
import struct
import binascii
from datetime import datetime
import warnings
import math
from math import floor
import importlib
import sys
import subprocess



logger = logging.getLogger(__name__)

# --- 辅助可视化函数：3D点云可视化 ---
def visualize_pcd_3d(points, title="3D Point Cloud", colors=None):
    """
    可视化3D点云。

    Args:
        points (np.array): 输入的点云数组，形状为 (N, 3)，N为点数，3为X,Y,Z坐标。
                           注意：此函数期望N,3的数组。
        title (str): 可视化窗口的标题。
        colors (np.array, optional): 点云的颜色数组，形状为 (N, 3)，RGB值在[0,1]之间。
                                     如果为None，则根据X坐标（通常代表深度）进行着色。
    """
    # 检查点云是否为空
    if len(points) == 0:
        print(f"警告: 无法可视化 '{title}'，因为点云为空。")
        return

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) # 将numpy数组转换为Open3D的Vector3dVector

    if colors is None:
        # 如果未提供颜色，则根据X坐标（深度）进行着色
        x_coords = points[:, 0]
        # 归一化X坐标到[0, 1]范围，以便应用于颜色映射
        # 避免除以零或非常小的数，以防所有X坐标相同
        if np.max(x_coords) - np.min(x_coords) > 1e-6:
            norm_x = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
        else:
            norm_x = np.zeros_like(x_coords) # 如果所有X坐标相同，则统一设置为0
        # 使用'viridis'颜色映射，并将结果的RGB部分作为点云颜色
        pcd.colors = o3d.utility.Vector3dVector(cm.get_cmap('viridis')(norm_x)[:,:3])
    else:
        # 如果提供了颜色，则直接使用
        pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"正在显示 '{title}' 的3D可视化...")
    # 绘制点云
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1024, height=768)


def visualize_clustered_pcd(points, labels, title="HDBSCAN Clustered Point Cloud"):
    """
    可视化DBSCAN聚类结果，不同聚类用不同颜色，噪声点用灰色。

    Args:
        points (np.array): 输入的点云数组，形状为 (N, 3)。
        labels (np.array): 聚类标签数组，形状为 (N,)，-1表示噪声点，其他非负整数表示聚类ID。
        title (str): 可视化窗口的标题。
    """
    # 检查点云是否为空
    if len(points) == 0:
        print(f"警告: 无法可视化 '{title}'，因为点云为空。")
        return

    # 初始化颜色数组，所有点默认颜色为黑色（或不设置，稍后会被覆盖）
    colors = np.zeros((points.shape[0], 3))
    # 获取所有独特的聚类标签
    unique_labels = np.unique(labels)

    # 获取颜色映射，例如tab20，它有20种不同的颜色，适合区分多个聚类
    cmap = cm.get_cmap('tab20')

    # 为每个聚类分配颜色
    for label_idx, label in enumerate(unique_labels):
        if label == -1: # 如果是噪声点
            colors[labels == label] = [0.5, 0.5, 0.5] # 分配灰色
        else:
            # 为每个有效聚类分配一个颜色
            # 使用取模运算以防聚类数量超过颜色映射的颜色数量，确保颜色循环使用
            cluster_color = cmap(label_idx % cmap.N)[:3]
            colors[labels == label] = cluster_color

    # 调用通用3D点云可视化函数显示带颜色的聚类结果
    visualize_pcd_3d(points, title=title, colors=colors)

# --- 2D投影可视化函数 ---
def visualize_yz_projection(points):
    """
    将3D点云投影到YZ平面，并使用X坐标作为颜色信息进行可视化。
    这有助于从俯视角度观察点云的深度信息，从而理解过滤后的点云形状。

    Args:
        points (np.array): 输入的点云数组，形状为 (N, 3)。
    """
    if len(points) == 0:
        print("警告: 无法可视化YZ投影，因为点云为空。")
        return

    # 提取Y坐标、Z坐标和X坐标（用于颜色映射）
    y_coords, z_coords, x_colors = points[:, 1], points[:, 2], points[:, 0]

    fig, ax = plt.subplots(figsize=(10, 10)) # 创建一个matplotlib图表

    # 绘制散点图，Y为X轴，Z为Y轴，颜色根据X坐标
    scatter = ax.scatter(y_coords, z_coords, s=1, c=x_colors, cmap='viridis')

    ax.set_xlabel("Y-axis (m)"); ax.set_ylabel("Z-axis (m)") # 设置轴标签
    ax.set_title("Filtered Point Cloud Projected onto YZ Plane (Colored by X-coordinate)") # 设置图表标题

    # 添加颜色条，解释X坐标与颜色的对应关系
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('X-coordinate (Depth in meters)')

    ax.set_aspect('equal', adjustable='box'); ax.grid(True) # 设置坐标轴比例相等，并显示网格

    plt.show() # 显示图表


# --- 基于高度和强度的粗过滤函数---
def coarse_filtration_point_cloud(
    initial_points: np.ndarray,
    x_initial_threshold: float = 10.0,
    x_stat_percentage: float = 0.5,
    x_filter_offset: float = 2.0,
    intensity_threshold: float = 10.0,
    x_upper_bound_adjustment: float = 1.5,
    visualize_coarse_filtration: bool = False
) -> np.ndarray:
    """
    对点云进行预处理，包括基于X坐标的初步过滤，以及基于X坐标（深度）和强度的组合过滤。

    参数:
        initial_points (np.ndarray): 原始点云数据，形状为 (N, 4)，假设列为 [X, Y, Z, Intensity]。
        x_initial_threshold (float): 第一个过滤步骤的X坐标阈值，只保留X >= 此值的点。
        x_stat_percentage (float): 用于统计X坐标平均值的点数比例。选择X坐标值最大的N%的点来确定基准X。
        x_filter_offset (float): X坐标强度过滤范围的偏移量（米）。
        intensity_threshold (float): 强度过滤阈值，只保留强度 >= 此值的点。
        x_upper_bound_adjustment (float): 基于平均X坐标调整X坐标上限的距离（米）。
                                         “平均X坐标的往上此值米以下的都不要” 意味着我们将此值从平均X减去，
                                         作为X坐标上限的基准。
        visualize_coarse_filtration (bool): 是否在中间步骤显示3D可视化。

    返回:
        np.ndarray: 经过过滤后的点云数据，形状为 (M, 4)，如果过滤后点云为空则返回空数组。
    """
    if initial_points.shape[1] < 4:
        logging.error("输入点云的维度不足4 (需要X, Y, Z, Intensity)。")
        return np.array([])
    if initial_points.ndim != 2:
        logging.error("输入点云的维度不正确，应为二维数组 (N, 4)。")
        return np.array([])

    points = initial_points.copy() # 避免修改原始输入数据

    # --- 1. 初步过滤: 基于X坐标移除X小于某个阈值的点 ---
    
    if len(points) == 0:
        logging.error("输入点云为空，跳过初步X坐标过滤。")
        return np.array([])



    x_filter_mask = points[:, 0] >= x_initial_threshold
    points = points[x_filter_mask]

    if len(points) == 0:
        logging.error("初步X坐标过滤后点云为空，无法进行后续处理。")
        return np.array([])


    # 去除X坐标最大的40%的点
    remove_count = int(len(points) * 0.4)
    
    if remove_count > 0:
        # 使用argpartition找到X坐标最大的20%的点的索引
        largest_x_indices = np.argpartition(points[:, 0], -remove_count)[-remove_count:]
        
        # 创建掩码，保留不在最大40%中的点
        keep_mask = np.ones(len(points), dtype=bool)
        keep_mask[largest_x_indices] = False
        
        # 应用掩码过滤点云，保留60%的点
        points = points[keep_mask]
    # --- 2. 预过滤: 基于X坐标（深度）和强度 ---
    # 统计X坐标最大的 X_STAT_PERCENTAGE 比例个点的平均值作为基准X (avg_largest_x)
    X_STAT_COUNT = int(len(points) * x_stat_percentage)  


    # 使用argpartition高效找到最大的N个值
    largest_x_indices = np.argpartition(points[:, 0], -X_STAT_COUNT)[-X_STAT_COUNT:]
    largest_x_values = points[largest_x_indices, 0]
    avg_largest_x = np.mean(largest_x_values)

    # 根据基准X和偏移量定义X过滤范围
    # "平均x坐标的往上1.5米以下的都不要" 意味着 X 值小于 (avg_largest_x - x_upper_bound_adjustment) 的区域是我们要处理的。
    # 所以 `x_upper_bound` 是这个过滤范围的远端（较大的X值）。
    x_upper_bound = avg_largest_x - x_upper_bound_adjustment # 大于x_upper_bound的都不要
    x_lower_bound = x_upper_bound - x_filter_offset          # 在x_upper_bound和x_lower_bound之间进行强度过滤

    # 2.1. 保留X坐标小于下限的点
    mask_too_close = (points[:, 0] < x_lower_bound)

    # 2.2. 在中间X范围 [x_lower_bound, x_upper_bound] 内，同时满足强度阈值的点
    mask_in_range_and_intense = (points[:, 0] >= x_lower_bound) & \
                                (points[:, 0] <= x_upper_bound) & \
                                (points[:, 3] >= intensity_threshold) # points[:, 3] 是强度

    # 将以上两种情况的点合并：满足任一条件即可保留
    final_combined_mask = mask_too_close | mask_in_range_and_intense
    final_filtered_points = points[final_combined_mask]


    # 3. 可视化最终过滤后的结果
    if len(final_filtered_points) > 0 and visualize_coarse_filtration:
        # 仅传递X, Y, Z用于可视化
        visualize_pcd_3d(final_filtered_points[:, :3], title="X-Coordinate & Intensity Pre-Filtered Point Cloud")
    elif len(final_filtered_points) == 0:
        logging.error("最终X坐标和强度预过滤后点云为空，无法进行后续处理。")
        return np.array([]) # 如果过滤后为空，则提前返回

    return final_filtered_points




def cluster_points_dbscan(points, eps=0.6, min_samples=5, visualize_dbscan=False):
    """
    
    Args:
        points (np.array): 点云数组 (N, 3)
        eps_equivalent (float): 等效的eps值
        min_samples (int): 最小样本数
        visualize_dbscan (bool): 是否可视化DBSCAN聚类结果
    Returns:
        np.array: 聚类标签
    """
    if len(points) == 0:
        logging.error("输入点云为空，跳过DBSCAN聚类。")
        return np.array([])
        
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    start_time = time.time()
    labels = clusterer.fit_predict(points.astype(np.float32))
    end_time = time.time()
    
    # 统计结果
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    #统计有多少噪音点
    n_noise = np.sum(labels == -1)
    

    if visualize_dbscan:
        visualize_clustered_pcd(points, labels, 
                              title=f"DBSCAN聚类 (eps={eps}, {n_clusters}个聚类)")
    return labels



def find_rectangle_by_histogram_method(filtered_points,pixel_size,kernel_size,white_threshold,radar_center_y,radar_center_z,is_first_time=False, visualize_steps=False,):
    """
    基于栅格化和直方图的矩形检测方法
    
    参数:
    - filtered_points: 过滤后的点云数据 (N, 3) [X, Y, Z]
    - visualize_steps: 是否显示中间步骤的可视化
    
    返回:
    - rectangle_corners_yz: 矩形在YZ平面的四个顶点 (4, 2) [Y, Z]
    """
    # 设置matplotlib中文字体，解决中文显示乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    if len(filtered_points) < 4:
        # print("点云数据不足，无法检测矩形")
        return None
    
    # 提取YZ投影点
    yz_points = filtered_points[:, [1, 2]]  # Y, Z坐标
    

    
    # 栅格化参数
    pixel_size = pixel_size  # 栅格分辨率，即每个像素代表的物理尺寸（米/像素）
    
    # 获取YZ坐标的最小值，用于将物理坐标映射到像素坐标的原点
    min_yz = np.min(yz_points, axis=0)
    
    # 将物理坐标转换为像素坐标（向量化操作）
    indices = ((yz_points - min_yz) / pixel_size).astype(int)
    
    # 计算图像的尺寸，取最大像素坐标+1
    grid_shape = np.max(indices, axis=0) + 1
    
    # 创建一个空的二值图像，Z坐标对应行，Y坐标对应列
    image_raw = np.zeros((grid_shape[1], grid_shape[0]), dtype=np.uint8)
    
    # 检查索引是否在图像范围内，防止越界
    valid_indices_mask = (indices[:, 1] < grid_shape[1]) & (indices[:, 0] < grid_shape[0]) & \
                         (indices[:, 1] >= 0) & (indices[:, 0] >= 0)
    
    # 标记有点的像素为255（白色）
    image_raw[indices[valid_indices_mask, 1], indices[valid_indices_mask, 0]] = 255
    
    if visualize_steps:
        plt.figure(figsize=(10, 10))
        plt.imshow(image_raw, cmap='gray')
        plt.title("步骤 1: 原始栅格化图像 (YZ投影)")
        plt.xlabel('Y轴 (像素)')
        plt.ylabel('Z轴 (像素)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    
    # 形态学核的大小，这个值决定了能填充多大的空洞和连接多远的间隙
    kernel_size = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 执行闭运算（膨胀 + 腐蚀）
    binary_image = cv2.morphologyEx(image_raw, cv2.MORPH_CLOSE, kernel)
    
    if visualize_steps:
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_image, cmap='gray')
        plt.title(f"步骤 2: 形态学闭运算后 (核大小 {kernel_size}x{kernel_size})")
        plt.xlabel('Y轴 (像素)')
        plt.ylabel('Z轴 (像素)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # 获取栅格图像的基本信息
    grid_height, grid_width = binary_image.shape
    
        # 确保中心点在图像范围内
    def clamp_point(y, z, width, height):
        return max(0, min(y, width - 1)), max(0, min(z, height - 1))
    # 迭代搜索参数
    min_rectangle_size = 15.0  # 最小矩形尺寸（米）
    max_iterations = 2 # 最大迭代次数，防止无限循环
    current_center = None  # 当前搜索中心点
    iteration = 0  # 迭代次数
    
    while iteration < max_iterations:
        iteration += 1
        # print(f"\n=== 第 {iteration} 次矩形检测 ===")

        if current_center is None and is_first_time==True:

            ###第一种使用(0,0)作为中心点#####
            #   # 第一次迭代：使用Y轴和Z轴的交点(0,0)作为中心点
            # center_pixel_y = int((0 - min_yz[0]) / pixel_size)
            # center_pixel_z = int((0 - min_yz[1]) / pixel_size)
            
            # # 确保中心点在图像范围内
            # center_pixel_y, center_pixel_z = clamp_point(center_pixel_y, center_pixel_z, grid_width, grid_height)


            #####第二种统计的方法找到中心点#####
            # 第一次迭代：从原点(0,0)向两端延长15米，统计黑色像素坐标的平均值
            # def calculate_center_from_origin():
            #     """
            #     从原点(0,0)向Y轴和Z轴方向延长15米，统计黑色像素坐标的平均值
            #     """
            #     # 将原点(0,0)转换为像素坐标
            #     origin_pixel_y = int((0 - min_yz[0]) / pixel_size)
            #     origin_pixel_z = int((0 - min_yz[1]) / pixel_size)
                
            #     # 15米对应的像素数
            #     extend_pixels = int(15.0 / pixel_size)
                
            #     # 收集Y轴方向上的黑色像素坐标
            #     y_black_pixels = []
            #     # Y轴正方向
            #     for i in range(extend_pixels + 1):
            #         pixel_y = origin_pixel_y + i
            #         pixel_z = origin_pixel_z
            #         if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
            #             if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
            #                 # 转换回物理坐标
            #                 physical_y = min_yz[0] + pixel_y * pixel_size
            #                 y_black_pixels.append(physical_y)
                
            #     # Y轴负方向
            #     for i in range(1, extend_pixels + 1):
            #         pixel_y = origin_pixel_y - i
            #         pixel_z = origin_pixel_z
            #         if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
            #             if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
            #                 # 转换回物理坐标
            #                 physical_y = min_yz[0] + pixel_y * pixel_size
            #                 y_black_pixels.append(physical_y)
                
            #     # 收集Z轴方向上的黑色像素坐标
            #     z_black_pixels = []
            #     # Z轴正方向
            #     for i in range(extend_pixels + 1):
            #         pixel_y = origin_pixel_y
            #         pixel_z = origin_pixel_z + i
            #         if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
            #             if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
            #                 # 转换回物理坐标
            #                 physical_z = min_yz[1] + pixel_z * pixel_size
            #                 z_black_pixels.append(physical_z)
                
            #     # Z轴负方向
            #     for i in range(1, extend_pixels + 1):
            #         pixel_y = origin_pixel_y
            #         pixel_z = origin_pixel_z - i
            #         if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
            #             if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
            #                 # 转换回物理坐标
            #                 physical_z = min_yz[1] + pixel_z * pixel_size
            #                 z_black_pixels.append(physical_z)
                
            #     # 计算平均值作为中心点坐标
            #     if len(y_black_pixels) > 0:
            #         center_y = np.mean(y_black_pixels)
            #     else:
            #         center_y = 0  # 如果没有黑色像素，使用原点Y坐标
                
            #     if len(z_black_pixels) > 0:
            #         center_z = np.mean(z_black_pixels)
            #     else:
            #         center_z = 0  # 如果没有黑色像素，使用原点Z坐标
                
            #     # print(f"Y轴方向找到 {len(y_black_pixels)} 个黑色像素，平均Y坐标: {center_y:.3f}")
            #     # print(f"Z轴方向找到 {len(z_black_pixels)} 个黑色像素，平均Z坐标: {center_z:.3f}")
                
            #     return center_y, center_z
            
            # # 计算新的中心点
            # center_y, center_z = calculate_center_from_origin()
            
            # # 转换为像素坐标
            # center_pixel_y = int((center_y - min_yz[0]) / pixel_size)
            # center_pixel_z = int((center_z - min_yz[1]) / pixel_size)
            
            # # 确保中心点在图像范围内
            # center_pixel_y, center_pixel_z = clamp_point(center_pixel_y, center_pixel_z, grid_width, grid_height)
            
            # # 重新计算物理坐标（确保在范围内）
            # center_y = min_yz[0] + center_pixel_y * pixel_size
            # center_z = min_yz[1] + center_pixel_z * pixel_size


            #####第二种统计加聚类的方法找到中心点#####
            # 第一次迭代：从原点(0,0)向两端延长15米，统计黑色像素坐标的平均值
            def calculate_center_from_origin():
                """
                从原点(0,0)向Y轴和Z轴方向延长15米，统计黑色像素坐标的平均值
                对Y轴和Z轴分别进行一维聚类分析
                """
                # 将原点(0,0)转换为像素坐标
                origin_pixel_y = int((0 - min_yz[0]) / pixel_size)
                origin_pixel_z = int((0 - min_yz[1]) / pixel_size)
                
                # 15米对应的像素数
                extend_pixels = int(15.0 / pixel_size)
                
                # 收集Y轴方向上的黑色像素坐标
                y_black_pixels = []
                # Y轴正方向
                for i in range(extend_pixels + 1):
                    pixel_y = origin_pixel_y + i
                    pixel_z = origin_pixel_z
                    if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
                        if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
                            # 转换回物理坐标
                            physical_y = min_yz[0] + pixel_y * pixel_size
                            y_black_pixels.append(physical_y)
                
                # Y轴负方向
                for i in range(1, extend_pixels + 1):
                    pixel_y = origin_pixel_y - i
                    pixel_z = origin_pixel_z
                    if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
                        if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
                            # 转换回物理坐标
                            physical_y = min_yz[0] + pixel_y * pixel_size
                            y_black_pixels.append(physical_y)
                
                # 收集Z轴方向上的黑色像素坐标
                z_black_pixels = []
                # Z轴正方向
                for i in range(extend_pixels + 1):
                    pixel_y = origin_pixel_y
                    pixel_z = origin_pixel_z + i
                    if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
                        if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
                            # 转换回物理坐标
                            physical_z = min_yz[1] + pixel_z * pixel_size
                            z_black_pixels.append(physical_z)
                
                # Z轴负方向
                for i in range(1, extend_pixels + 1):
                    pixel_y = origin_pixel_y
                    pixel_z = origin_pixel_z - i
                    if (0 <= pixel_y < grid_width and 0 <= pixel_z < grid_height):
                        if binary_image[pixel_z, pixel_y] == 0:  # 黑色像素
                            # 转换回物理坐标
                            physical_z = min_yz[1] + pixel_z * pixel_size
                            z_black_pixels.append(physical_z)
                
                # print(f"Y轴方向找到 {len(y_black_pixels)} 个黑色像素")
                # print(f"Z轴方向找到 {len(z_black_pixels)} 个黑色像素")
                
                # 对Y轴坐标进行一维聚类
                def cluster_1d_coordinates(coordinates, axis_name):
                    """
                    对一维坐标进行聚类分析
                    """
                    if len(coordinates) == 0:
                        # print(f"{axis_name}轴: 没有找到黑色像素")
                        return 0, [], []
                    
                    # 转换为numpy数组并reshape为二维（DBSCAN需要）
                    coords_array = np.array(coordinates).reshape(-1, 1)
                    
                    # 使用DBSCAN进行一维聚类
                    eps_distance = 1.0  # 1米的聚类半径
                    min_samples = 2     # 最少2个点形成一个聚类
                    
                    clustering = DBSCAN(eps=eps_distance, min_samples=min_samples)
                    cluster_labels = clustering.fit_predict(coords_array)
                    
                    # 统计聚类结果
                    unique_labels = np.unique(cluster_labels)
                    n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    # print(f"{axis_name}轴聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")
                    
                    # 分析每个聚类
                    cluster_info = []
                    for label in unique_labels:
                        if label == -1:  # 噪声点
                            continue
                        
                        cluster_mask = cluster_labels == label
                        cluster_coords = np.array(coordinates)[cluster_mask]
                        cluster_size = len(cluster_coords)
                        cluster_center = np.mean(cluster_coords)
                        cluster_std = np.std(cluster_coords) if cluster_size > 1 else 0
                        
                        cluster_info.append({
                            'label': label,
                            'size': cluster_size,
                            'center': cluster_center,
                            'std': cluster_std,
                            'coords': cluster_coords,
                            'min': float(np.min(cluster_coords)),  
                            'max': float(np.max(cluster_coords))
                        })
                        
                        # print(f"{axis_name}轴聚类 {label}: {cluster_size} 个点, 中心: {cluster_center:.3f}, 标准差: {cluster_std:.3f}, 范围: [{np.min(cluster_coords):.3f}, {np.max(cluster_coords):.3f}]")
                    
                   
                    if len(cluster_info) == 0:
                        # print(f"{axis_name}轴: 没有有效聚类，使用所有点的平均值")
                        final_center = np.mean(coordinates)
                    else:
                        # 按聚类大小排序
                        cluster_info.sort(key=lambda x: x['size'], reverse=True)
                        largest_cluster = cluster_info[0]
                        final_center = largest_cluster['center']
                        # print(f"{axis_name}轴: 选择最大聚类 {largest_cluster['label']} (包含 {largest_cluster['size']} 个点)")
                    
                    return final_center, cluster_info, cluster_labels
                
                # 分别对Y轴和Z轴进行聚类
                center_y, y_cluster_info, y_labels = cluster_1d_coordinates(y_black_pixels, "Y")
                center_z, z_cluster_info, z_labels = cluster_1d_coordinates(z_black_pixels, "Z")
                
                # 对Y轴和Z轴分别进行聚类筛选
                def filter_clusters_by_continuity(cluster_info, axis_name):
                    """
                    基于连续性筛选聚类：
                    1. 选择距离原点(0)最近的聚类作为主聚类
                    2. 向左右两端扩展，如果相邻聚类边界距离超过4米则停止扩展
                    3. 如果相邻聚类边界距离小于4米，则将其作为新的主聚类继续扩展
                    """
                    if len(cluster_info) == 0:
                        # print(f"{axis_name}轴: 没有聚类可供筛选")
                        return [], 0
                    
                    # 按聚类中心坐标排序
                    sorted_clusters = sorted(cluster_info, key=lambda x: x['center'])
                    
                    # print(f"\n{axis_name}轴聚类筛选过程:")
                    # print(f"原始聚类数量: {len(sorted_clusters)}")
                    # for i, cluster in enumerate(sorted_clusters):
                    #     print(f"  聚类 {cluster['label']}: 中心={cluster['center']:.3f}, 范围=[{cluster['min']:.3f}, {cluster['max']:.3f}], 大小={cluster['size']}")
                    
                # --- 找到聚类中包含距离原点最近的点的聚类作为主聚类 ---
                    min_dist_to_origin = float('inf')
                    main_cluster = None
                    main_cluster_idx = -1
                    
                    # 遍历所有聚类，找到所有点中距离原点最近的那个点，并确定其所属的聚类
                    for i, cluster in enumerate(sorted_clusters):
                        for point in cluster['coords']:
                            dist = abs(point) # 点到原点(0)的距离
                            if dist < min_dist_to_origin:
                                min_dist_to_origin = dist
                                main_cluster = cluster
                                main_cluster_idx = i # 记录包含最近点的聚类的索引
                    
                    # print(f"选择距离原点最近的聚类作为主聚类: 聚类 {main_cluster['label']} (中心={main_cluster['center']:.3f}, 距离原点={distances_to_origin[main_cluster_idx]:.3f})")
                    
                    # 初始化有效聚类列表
                    valid_clusters = [main_cluster]
                    current_main_idx = main_cluster_idx
                    
                    # 向左端扩展
                    # print(f"\n向左端扩展:")
                    left_main_idx = current_main_idx
                    while left_main_idx > 0:
                        current_cluster = sorted_clusters[left_main_idx]
                        left_cluster = sorted_clusters[left_main_idx - 1]
                        
                        # 计算边界距离：主聚类的最左端到左端聚类的最右端
                        main_left_boundary = current_cluster['min']
                        left_right_boundary = left_cluster['max']
    
                        distance = abs(main_left_boundary - left_right_boundary)
                        
  
                        
                        if distance >= 2.0:
                            # print(f"  边界距离 {distance:.3f} >= 2.0米，停止左端扩展")
                            break
                        else:
                            # print(f"  边界距离 {distance:.3f} < 2.0米，保留聚类 {left_cluster['label']} 并将其设为新的主聚类")
                            valid_clusters.insert(0, left_cluster)  # 插入到列表开头
                            left_main_idx -= 1  # 更新主聚类索引
                    
                    # 向右端扩展
                    # print(f"\n向右端扩展:")
                    right_main_idx = current_main_idx
                    while right_main_idx < len(sorted_clusters) - 1:
                        current_cluster = sorted_clusters[right_main_idx]
                        right_cluster = sorted_clusters[right_main_idx + 1]
                        
                        # 计算边界距离：主聚类的最右端到右端聚类的最左端
                        main_right_boundary = current_cluster['max']
                        right_left_boundary = right_cluster['min']
                        distance = abs(right_left_boundary - main_right_boundary)
                        
                        # print(f"  检查聚类 {right_cluster['label']} (范围=[{right_cluster['min']:.3f}, {right_cluster['max']:.3f}]) 与当前主聚类 {current_cluster['label']} (范围=[{current_cluster['min']:.3f}, {current_cluster['max']:.3f}])")
                        # print(f"  边界距离: 右聚类左端({right_left_boundary:.3f}) - 主聚类右端({main_right_boundary:.3f}) = {distance:.3f} 米")
                        
                        if distance >= 2.0:
                            # print(f"  边界距离 {distance:.3f} >= 2.0米，停止右端扩展")
                            break
                        else:
                            # print(f"  边界距离 {distance:.3f} < 2.0米，保留聚类 {right_cluster['label']} 并将其设为新的主聚类")
                            valid_clusters.append(right_cluster)  # 添加到列表末尾
                            right_main_idx += 1  # 更新主聚类索引
                    
                    # print(f"\n{axis_name}轴筛选结果:")
                    # print(f"保留的聚类数量: {len(valid_clusters)}")
                    
                    if len(valid_clusters) == 0:
                        return [], 0
                    
                    # 计算加权中心
                    total_weight = sum(cluster['size'] for cluster in valid_clusters)
                    weighted_center = sum(cluster['center'] * cluster['size'] for cluster in valid_clusters) / total_weight

                    
                    return valid_clusters, weighted_center
                
                # 应用连续性筛选
                # print(f"\n=== 开始连续性筛选 ===")
                valid_y_clusters, filtered_center_y = filter_clusters_by_continuity(y_cluster_info, "Y")
                valid_z_clusters, filtered_center_z = filter_clusters_by_continuity(z_cluster_info, "Z")
                
                # 更新最终中心点
                if len(valid_y_clusters) > 0:
                    center_y = filtered_center_y
                if len(valid_z_clusters) > 0:
                    center_z = filtered_center_z
                
                # 可视化筛选后的聚类结果
                def visualize_filtered_clustering():
                    """
                    可视化筛选后的Y轴和Z轴聚类结果
                    """
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
                    
                    # Y轴聚类可视化
                    if len(y_black_pixels) > 0:
                        # 绘制所有原始聚类（灰色，半透明）
                        for cluster in y_cluster_info:
                            coords = cluster['coords']
                            y_positions = np.random.normal(-0.3, 0.05, len(coords))
                            ax1.scatter(coords, y_positions, 
                                      c='lightgray', s=20, alpha=0.3,
                                      label='原始聚类' if cluster == y_cluster_info[0] else "")
                        
                        # 绘制筛选后的有效聚类（彩色）
                        y_colors = plt.cm.Set1(np.linspace(0, 1, len(valid_y_clusters) + 1))
                        for i, cluster in enumerate(valid_y_clusters):
                            coords = cluster['coords']
                            y_positions = np.random.normal(0.3, 0.05, len(coords))
                            ax1.scatter(coords, y_positions, 
                                      c=[y_colors[i]], 
                                      label=f'有效聚类 {cluster["label"]} ({cluster["size"]}点)',
                                      s=60, alpha=0.8)
                            
                            # 标记聚类中心
                            ax1.axvline(x=cluster['center'], color=y_colors[i], 
                                      linestyle='--', alpha=0.8, linewidth=2)
                            ax1.text(cluster['center'], 0.6, f'C{cluster["label"]}', 
                                   rotation=90, ha='center', va='bottom', fontweight='bold')
                        
                        # 标记最终中心和原点
                        ax1.axvline(x=center_y, color='red', linestyle='-', linewidth=4, 
                                  label=f'最终中心 ({center_y:.3f})')
                        ax1.axvline(x=0, color='black', linestyle='-', linewidth=3, 
                                  label='原点 (0)')
                        
                        ax1.set_xlabel('Y坐标 (米)')
                        ax1.set_ylabel('显示层级')
                        ax1.set_title('Y轴聚类连续性筛选结果')
                        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax1.grid(True, alpha=0.3)
                        ax1.set_ylim(-0.8, 0.8)
                    
                    # Z轴聚类可视化
                    if len(z_black_pixels) > 0:
                        # 绘制所有原始聚类（灰色，半透明）
                        for cluster in z_cluster_info:
                            coords = cluster['coords']
                            y_positions = np.random.normal(-0.3, 0.05, len(coords))
                            ax2.scatter(coords, y_positions, 
                                      c='lightgray', s=20, alpha=0.3,
                                      label='原始聚类' if cluster == z_cluster_info[0] else "")
                        
                        # 绘制筛选后的有效聚类（彩色）
                        z_colors = plt.cm.Set2(np.linspace(0, 1, len(valid_z_clusters) + 1))
                        for i, cluster in enumerate(valid_z_clusters):
                            coords = cluster['coords']
                            y_positions = np.random.normal(0.3, 0.05, len(coords))
                            ax2.scatter(coords, y_positions, 
                                      c=[z_colors[i]], 
                                      label=f'有效聚类 {cluster["label"]} ({cluster["size"]}点)',
                                      s=60, alpha=0.8)
                            
                            # 标记聚类中心
                            ax2.axvline(x=cluster['center'], color=z_colors[i], 
                                      linestyle='--', alpha=0.8, linewidth=2)
                            ax2.text(cluster['center'], 0.6, f'C{cluster["label"]}', 
                                   rotation=90, ha='center', va='bottom', fontweight='bold')
                        
                        # 标记最终中心和原点
                        ax2.axvline(x=center_z, color='red', linestyle='-', linewidth=4, 
                                  label=f'最终中心 ({center_z:.3f})')
                        ax2.axvline(x=0, color='black', linestyle='-', linewidth=3, 
                                  label='原点 (0)')
                        
                        ax2.set_xlabel('Z坐标 (米)')
                        ax2.set_ylabel('显示层级')
                        ax2.set_title('Z轴聚类连续性筛选结果')
                        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax2.grid(True, alpha=0.3)
                        ax2.set_ylim(-0.8, 0.8)
                    
                    plt.tight_layout()
                    plt.show()
                
                # 调用可视化函数
                if visualize_steps:
                    visualize_filtered_clustering()
                

                
                return center_y, center_z
            
            # 计算新的中心点
            center_y, center_z = calculate_center_from_origin()
            
            # 转换为像素坐标
            center_pixel_y = int((center_y - min_yz[0]) / pixel_size)
            center_pixel_z = int((center_z - min_yz[1]) / pixel_size)
            
            # 确保中心点在图像范围内
            center_pixel_y, center_pixel_z = clamp_point(center_pixel_y, center_pixel_z, grid_width, grid_height)
            
            # 重新计算物理坐标（确保在范围内）
            center_y = min_yz[0] + center_pixel_y * pixel_size
            center_z = min_yz[1] + center_pixel_z * pixel_size

        elif  current_center is None and is_first_time==False:
            #使用真实船舱中心转换成雷达坐标。
            center_pixel_y = int((radar_center_y - min_yz[0]) / pixel_size)
            center_pixel_z = int((radar_center_z - min_yz[1]) / pixel_size)

              # 确保中心点在图像范围内
            center_pixel_y, center_pixel_z = clamp_point(center_pixel_y, center_pixel_z, grid_width, grid_height)
            # 重新计算物理坐标（确保在范围内）
            center_y = min_yz[0] + center_pixel_y * pixel_size
            center_z = min_yz[1] + center_pixel_z * pixel_size

        else:
            # 后续迭代：使用上次检测到的矩形中心点
            center_pixel_y = int((current_center[0] - min_yz[0]) / pixel_size)
            center_pixel_z = int((current_center[1] - min_yz[1]) / pixel_size)
            
            # 确保中心点在图像范围内
            center_pixel_y, center_pixel_z = clamp_point(center_pixel_y, center_pixel_z, grid_width, grid_height)
            
            # 转换为物理坐标
            center_y = min_yz[0] + center_pixel_y * pixel_size
            center_z = min_yz[1] + center_pixel_z * pixel_size
        
        center_point = np.array([center_y, center_z])
        

        
                # 步骤4: 从中心点向四个方向扫描，寻找边界
        def scan_direction_for_edge(image, center_pixel, direction):
            """
            从中心点向指定方向扫描，寻找边界
            
            参数:
            - image: 二值图像
            - center_pixel: 中心点像素坐标 [y, z]
            - direction: 扫描方向向量 [dy, dz]
            
            返回:
            - edge_distance_pixels: 检测到的边缘距离（像素）
            """
            direction = np.array(direction, dtype=float)
            direction = direction / np.linalg.norm(direction)  # 归一化
            
            # 根据扫描方向确定扫描线长度
            if abs(direction[0]) > abs(direction[1]):  # Y轴方向（水平方向）
                scan_line_length = int(14.0 / pixel_size)  # Y轴方向30米
            else:  # Z轴方向（垂直方向）
                scan_line_length = int(10.0 / pixel_size)  # Z轴方向20米
            
            # 计算最大扫描距离（到图像边界）
            max_scan_distance = min(image.shape[0], image.shape[1])  # 使用图像的最小尺寸作为最大扫描距离
            
            for dist in range(1, max_scan_distance):
                # 计算扫描线中心点
                scan_center = np.array(center_pixel) + direction * dist
                scan_center_y, scan_center_z = int(scan_center[0]), int(scan_center[1])
                
                # 检查是否超出图像边界
                if (scan_center_y < 0 or scan_center_y >= image.shape[1] or 
                    scan_center_z < 0 or scan_center_z >= image.shape[0]):
                    # 如果到达边界还没找到边缘，返回边界距离
                    return dist - 1
                
                # 计算扫描线的垂直方向
                perpendicular = np.array([-direction[1], direction[0]])
                
                # 在扫描线上采样点
                white_pixel_count = 0
                total_pixel_count = 0
                
                for offset in range(-scan_line_length, scan_line_length + 1):
                    sample_point = scan_center + perpendicular * offset
                    sample_y, sample_z = int(sample_point[0]), int(sample_point[1])
                    
                    # 检查采样点是否在图像范围内
                    if (0 <= sample_y < image.shape[1] and 0 <= sample_z < image.shape[0]):
                        total_pixel_count += 1
                        if image[sample_z, sample_y] == 255:  # 白色像素（有点云数据）
                            white_pixel_count += 1
                
                # 计算白色像素比例
                if total_pixel_count > 0:
                    white_ratio = white_pixel_count / total_pixel_count
                    
                    # 当白色像素比例达到35%时，认为找到了边界
                    if white_ratio >= white_threshold:
                        return dist
            
            # 如果扫描到图像边界都没找到边缘，返回最大扫描距离
            return max_scan_distance - 1
        
        # 四个扫描方向
        directions = {
            'positive_y': [1, 0],   # Y轴正方向
            'negative_y': [-1, 0],  # Y轴负方向
            'positive_z': [0, 1],   # Z轴正方向
            'negative_z': [0, -1]   # Z轴负方向
        }
        
        # 扫描四个方向，获取边界距离
        edge_distances_pixels = {}
        for direction_name, direction_vector in directions.items():
            edge_distance_pixels = scan_direction_for_edge(binary_image, [center_pixel_y, center_pixel_z], direction_vector)
            edge_distances_pixels[direction_name] = edge_distance_pixels
            edge_distance_physical = edge_distance_pixels * pixel_size
            # print(f"{direction_name}方向边界距离: {edge_distance_pixels}像素 ({edge_distance_physical:.2f}米)")
        
        # 计算矩形的四个边界坐标
        edge_y_positive = center_y + edge_distances_pixels['positive_y'] * pixel_size
        edge_y_negative = center_y - edge_distances_pixels['negative_y'] * pixel_size
        edge_z_positive = center_z + edge_distances_pixels['positive_z'] * pixel_size
        edge_z_negative = center_z - edge_distances_pixels['negative_z'] * pixel_size
        
        # 构建矩形的四个角点
        rectangle_corners_yz = np.array([
            [edge_y_negative, edge_z_negative],  # 左下角
            [edge_y_positive, edge_z_negative],  # 右下角
            [edge_y_positive, edge_z_positive],  # 右上角
            [edge_y_negative, edge_z_positive]   # 左上角
        ])
        
        # 计算矩形的尺寸和中心点
        rectangle_width = edge_y_positive - edge_y_negative
        rectangle_height = edge_z_positive - edge_z_negative
        rectangle_center_y = (edge_y_positive + edge_y_negative) / 2
        rectangle_center_z = (edge_z_positive + edge_z_negative) / 2
        

        
        # 每次迭代都更新中心点为当前检测到的矩形中心点
        current_center = np.array([rectangle_center_y, rectangle_center_z])
        # print(f"更新搜索中心点为矩形中心: Y={rectangle_center_y:.3f}, Z={rectangle_center_z:.3f}")
        

        if iteration >= 1:  
            # 检查矩形大小是否满足要求
            if min_rectangle_size <= rectangle_width<=20 and min_rectangle_size <= rectangle_height<=30:
                # print(f"\n✓ 矩形尺寸满足要求 ({rectangle_width:.2f}m × {rectangle_height:.2f}m >= {min_rectangle_size}m × {min_rectangle_size}m)")
                
                if visualize_steps:
                    # 显示最终结果
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    
                    # 显示栅格化的二值图像
                    ax1.imshow(binary_image, cmap='gray', origin='lower', 
                               extent=[min_yz[0], min_yz[0] + grid_width * pixel_size, 
                                       min_yz[1], min_yz[1] + grid_height * pixel_size])
                    ax1.plot(center_y, center_z, 'ro', markersize=8, label='搜索中心点')
                    ax1.plot(rectangle_center_y, rectangle_center_z, 'go', markersize=8, label='矩形中心点')
                    ax1.set_xlabel('Y坐标 (米)')
                    ax1.set_ylabel('Z坐标 (米)')
                    ax1.set_title(f'第{iteration}次检测 - 栅格化二值图像')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # 显示检测结果
                    ax2.scatter(yz_points[:, 0], yz_points[:, 1], c='blue', s=1, alpha=0.5, label='原始点云')
                    
                    # 绘制检测到的矩形
                    rect_plot = np.vstack([rectangle_corners_yz, rectangle_corners_yz[0]])  # 闭合矩形
                    ax2.plot(rect_plot[:, 0], rect_plot[:, 1], 'r-', linewidth=2, label='检测到的矩形')
                    ax2.plot(center_y, center_z, 'ro', markersize=8, label='搜索中心点')
                    ax2.plot(rectangle_center_y, rectangle_center_z, 'go', markersize=8, label='矩形中心点')
                    
                    ax2.set_xlabel('Y坐标 (米)')
                    ax2.set_ylabel('Z坐标 (米)')
                    ax2.set_title(f'第{iteration}次检测 - 矩形检测结果 ({rectangle_width:.1f}m × {rectangle_height:.1f}m)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_aspect('equal')
                    
                    plt.tight_layout()
                    plt.show()
                
                return rectangle_corners_yz

        
        if visualize_steps and (iteration < 2 or (rectangle_width < min_rectangle_size or rectangle_height < min_rectangle_size)):
            # 显示当前迭代的结果
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.scatter(yz_points[:, 0], yz_points[:, 1], c='blue', s=1, alpha=0.5, label='原始点云')
            
            # 绘制检测到的矩形
            rect_plot = np.vstack([rectangle_corners_yz, rectangle_corners_yz[0]])  # 闭合矩形
            ax.plot(rect_plot[:, 0], rect_plot[:, 1], 'r-', linewidth=2, 
                   label=f'第{iteration}次检测矩形 ({rectangle_width:.1f}m × {rectangle_height:.1f}m)')
            ax.plot(center_y, center_z, 'ro', markersize=8, label='搜索中心点')
            ax.plot(rectangle_center_y, rectangle_center_z, 'go', markersize=8, label='矩形中心点')
            
            ax.set_xlabel('Y坐标 (米)')
            ax.set_ylabel('Z坐标 (米)')
            if iteration < 2:
                ax.set_title(f'第{iteration}次检测 - 强制迭代')
            else:
                ax.set_title(f'第{iteration}次检测 - 矩形过小，继续搜索')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.show()

    

    logger.error("find_rectangle_in_yz_plane: 未找到满足条件的矩形")
    return None

# def refine_x_coordinates_by_advanced_search(full_original_points_with_intensity, points_for_yz_refinement, refined_corners_yz, visualize_ranges=True):
#     """
#     通过在矩形短边周围的特定区域搜寻原始点云，使用高精度方法来精确化矩形顶点的X坐标。
    
#     功能概述：
#     1. 构建KD-tree索引以优化点云查询性能
#     2. 计算矩形各边长度，识别短边（船舱的宽度方向）
#     3. 在短边端点附近定义精细搜索区域
#     4. 使用统计方法计算每个区域的最优X坐标
#     5. 将计算结果分配给对应的矩形顶点
    
    
#     参数:
#         full_original_points_with_intensity: 完整的原始点云数据（包含强度信息）
#         points_for_yz_refinement: 用于YZ平面精炼的点云数据
#         refined_corners_yz: 已精炼的矩形角点YZ坐标
#         visualize_ranges: 是否可视化搜索范围
    
#     返回:
#         refined_corners_3d: 精炼后的3D矩形角点坐标
#         search_geometries: 搜索区域的可视化几何体
#         []: 空列表（保持接口兼容性）
#     """
  
#     # print("\n--- 开始高精度X坐标精确化（增强KD-tree版本）---")
    
#     # ==================== 步骤1: 构建KD-tree索引 ====================
#     # 构建2D KD-tree用于YZ平面的高效空间查询
#     # KD-tree将O(n)的线性搜索优化为O(log n)的树搜索
#     # print("构建2D KD-tree索引（YZ平面）...")
#     points_yz = full_original_points_with_intensity[:, 1:3]  # 提取YZ坐标
#     kdtree_yz = KDTree(points_yz)
#     # print(f"2D KD-tree构建完成，包含 {len(points_yz)} 个点")

#     # ==================== 步骤2: 计算边长度和存储边信息 ====================
#     # 计算矩形四条边的长度，为后续短边识别做准备
#     edge_lengths = []
#     edges = []
#     for i in range(4):
#         p1 = refined_corners_yz[i]
#         p2 = refined_corners_yz[(i + 1) % 4]  # 循环连接形成闭合矩形
#         edge_length = np.linalg.norm(p2 - p1)  # 计算欧几里得距离
#         edge_lengths.append(edge_length)
#         edges.append((p1, p2))

#     # print(f"矩形四条边长度: {[f'{l:.3f}' for l in edge_lengths]}")

#     # ==================== 步骤3: 计算基准X坐标 ====================
#     # 使用统计方法计算基准X坐标，避免极值点的影响
#     # 选择中间20%的点（60%-80%分位数）来计算更稳定的基准值
#     if len(points_for_yz_refinement) > 0:
#         all_x_values = points_for_yz_refinement[:, 0]
#         sorted_x_values = np.sort(all_x_values)
        
#         total_points = len(sorted_x_values)
#         start_idx = int(total_points * 0.6)  # 60%分位数
#         end_idx = int(total_points * 0.8)    # 80%分位数
        
#         if start_idx < end_idx:
#             # 使用中间20%的点计算基准值，排除极值影响
#             middle_x_values = sorted_x_values[start_idx:end_idx]
#             avg_x_baseline = np.mean(middle_x_values)
#             # print(f"使用过滤后点云中X坐标中间20%的点计算基准x: {avg_x_baseline:.3f} 米")
#         else:
#             # 点数太少时使用全部点
#             avg_x_baseline = np.mean(all_x_values)
#             # print(f"使用所有过滤后点计算基准x: {avg_x_baseline:.3f} 米")
#     else:
#         avg_x_baseline = 25
#         # print("警告: 没有可用的过滤后点来计算基准x，使用默认值 25")

#     # ==================== 步骤4: 对每条边进行X坐标采样 ====================
#     # 使用KD-tree优化的边缘采样，为短边识别提供数据支持
#     edge_x_averages = []
    
#     # 搜索参数定义
#     inward_search_distance = 0.5   # 向矩形内部的搜索距离（米）
#     outward_search_distance = 1.5  # 向矩形外部的搜索距离（米）
#     x_sample_range = 5             # X坐标采样范围（米）
#     edge_sample_geometries = []
#     for i, (p1, p2) in enumerate(edges):
#         # 计算边的方向向量和垂直向量
#         edge_vec = p2 - p1
#         edge_length = np.linalg.norm(edge_vec)
#         if edge_length < 1e-6:  # 处理退化边的情况
#             edge_x_averages.append(avg_x_baseline)
#             logger.debug(f"边 {i}: 长度接近零，使用全局平均值: {avg_x_baseline:.3f}")
#             continue

#         edge_unit = edge_vec / edge_length  # 边的单位方向向量
#         perp_vec = np.array([-edge_unit[1], edge_unit[0]])  # 垂直向量（逆时针90度）
        
#         # 确定垂直向量指向矩形外部
#         rect_center = np.mean(refined_corners_yz, axis=0)
#         edge_center = (p1 + p2) / 2
#         to_center = rect_center - edge_center
#         if np.dot(perp_vec, to_center) > 0:
#             perp_vec = -perp_vec  # 翻转方向使其指向外部

#         # 使用KD-tree进行高效的范围查询
#         search_radius = max(outward_search_distance + inward_search_distance, edge_length / 2)
#         candidate_indices = kdtree_yz.query_radius([edge_center], r=search_radius)[0]
        
#         # 对候选点进行精确的几何筛选
#         edge_sample_points = []
#         for idx in candidate_indices:
#             point = full_original_points_with_intensity[idx]
#             point_yz = point[1:3]
#             point_x = point[0]

#             # 计算点到边起点的投影
#             to_point = point_yz - p1
#             proj_along_edge = np.dot(to_point, edge_unit)    # 沿边方向的投影
#             proj_perp_edge = np.dot(to_point, perp_vec)     # 垂直边方向的投影

#             # 多重条件筛选：边界内 + 搜索带内 + X坐标范围内
#             if (0 <= proj_along_edge <= edge_length and
#                 -inward_search_distance <= proj_perp_edge <= outward_search_distance and
#                 avg_x_baseline - x_sample_range <= point_x <= avg_x_baseline + x_sample_range):
#                 edge_sample_points.append(point_x)

#         # 创建边采样范围的可视化几何体
#         if visualize_ranges:
#             # 创建边采样区域的长方体
#             box_x_dim = x_sample_range * 2
#             box_y_dim = edge_length
#             box_z_dim = outward_search_distance + inward_search_distance
            
#             edge_box = o3d.geometry.TriangleMesh.create_box(
#                 width=box_x_dim,
#                 height=box_y_dim, 
#                 depth=box_z_dim
#             )
            
#             # 移动到原点
#             edge_box.translate([-box_x_dim/2, -box_y_dim/2, -box_z_dim/2])
            
#             # 创建旋转矩阵，使长方体沿着边的方向
#             R = np.identity(3)
#             # Y轴沿着边的方向
#             R[1, 1] = edge_unit[0]  # edge_unit的Y分量
#             R[2, 1] = edge_unit[1]  # edge_unit的Z分量
#             # Z轴沿着垂直方向
#             R[1, 2] = perp_vec[0]   # perp_vec的Y分量
#             R[2, 2] = perp_vec[1]   # perp_vec的Z分量
            
#             edge_box.rotate(R, center=(0,0,0))
            
#             # 移动到边的中心位置
#             box_yz_center = edge_center + perp_vec * (-inward_search_distance+outward_search_distance) / 2
#             final_box_center_3d = np.array([avg_x_baseline, box_yz_center[0], box_yz_center[1]])
            
#             edge_box.translate(final_box_center_3d)
            
#             # 为不同的边设置不同的颜色
#             colors = [[1.0, 0.0, 0.0],  # 红色
#                      [0.0, 1.0, 0.0],  # 绿色
#                      [0.0, 0.0, 1.0],  # 蓝色
#                      [1.0, 1.0, 0.0]]  # 黄色
#             edge_box.paint_uniform_color(colors[i])
#             edge_sample_geometries.append(edge_box)
            
#             # 创建边的中心线
#             line_start_3d = np.array([avg_x_baseline, p1[0], p1[1]])
#             line_end_3d = np.array([avg_x_baseline, p2[0], p2[1]])
            
#             line_points = [line_start_3d, line_end_3d]
#             line_indices = [[0, 1]]
#             line_set = o3d.geometry.LineSet(
#                 points=o3d.utility.Vector3dVector(line_points),
#                 lines=o3d.utility.Vector2iVector(line_indices)
#             )
#             line_set.paint_uniform_color(colors[i])
#             edge_sample_geometries.append(line_set)
#         # 统计采样结果并计算边的X坐标平均值
#         if edge_sample_points:
#             sorted_points = np.sort(edge_sample_points)
#             if len(sorted_points) >= 100:
#                 # 优先使用最小的50个点（更接近船舱边缘）
#                 selected_points = sorted_points[:50]
#                 edge_x_avg = np.mean(selected_points)
#                 # print(f"边 {i}: KD-tree查询到 {len(candidate_indices)} 个候选点，采样到 {len(edge_sample_points)} 个点，使用最小50个点平均x坐标: {edge_x_avg:.3f}")
#             else:
#                 # 点数不足时使用全部点
#                 edge_x_avg = np.mean(sorted_points)
#                 # print(f"边 {i}: KD-tree查询到 {len(candidate_indices)} 个候选点，采样到 {len(edge_sample_points)} 个点，使用所有点平均x坐标: {edge_x_avg:.3f}")
#         else:
#             # # 未找到有效点时使用全局基准值
#             # edge_x_avg = avg_x_baseline
#             # print(f"边 {i}: KD-tree查询到 {len(candidate_indices)} 个候选点，但未采样到符合条件的点，使用全局平均值: {avg_x_baseline:.3f}")
#              # 未找到有效点时使用全局基准值
#             edge_x_avg = 999
#             # print(f"边 {i}: KD-tree查询到 {len(candidate_indices)} 个候选点，但未采样到符合条件的点，使用999")

#         edge_x_averages.append(edge_x_avg)

#     # ==================== 步骤5: 识别短边 ====================
#     # X坐标值较大，舱盖就不在这条边上
#     sorted_indices = np.argsort(edge_x_averages)[::-1]  # 按X坐标降序排列
#     short_edge_indices = sorted_indices[:2]  # 选择X坐标最大的两条边

#     # 验证选择的边是否为相对边（矩形的对边）
#     valid_combinations = [{0, 2}, {1, 3}]  # 有效的对边组合
#     selected_set = set(short_edge_indices)

#     if selected_set not in valid_combinations:
#         # 如果不是对边，排除最小x坐标边及其对边，选择另一对边
#         min_x_edge = sorted_indices[-1]  # x坐标最小的边
#         min_x_opposite_edge = (min_x_edge + 2) % 4  # 最小x坐标边的对边
#         excluded_edges = {min_x_edge, min_x_opposite_edge}
        
#         # 选择剩余的另一对相对边
#         all_edges = {0, 1, 2, 3}
#         remaining_edges = all_edges - excluded_edges
#         short_edge_indices = np.array(list(remaining_edges))
#         # print(f"选出的边不是相对边，排除最小x坐标边 {min_x_edge} 及其对边 {min_x_opposite_edge}，选择另一对边: {short_edge_indices}")

#     # print(f"各边x坐标平均值: {[f'{avg:.3f}' for avg in edge_x_averages]}")
#     # print(f"最终选择的短边索引: {short_edge_indices}")

#     # ==================== 步骤6: 定义精细搜索区域 ====================
#     # 在短边的端点附近创建精细搜索区域，用于高精度X坐标计算
    
#     # 搜索区域参数
#     shorten_dist = 4       # 边缩短距离（米），避免角点处的噪声和角点处没有点的情况
#     z_expand_min = -0.4       # Z方向向内扩展距离（米）
#     z_expand_max = 0.8        # Z方向向外扩展距离（米）
#     search_rect_width = 1   # 搜索矩形宽度（米）
#     x_search_range = 5        # X方向搜索范围（米）

#     search_regions = []       # 存储搜索区域信息
#     search_geometries = []    # 存储可视化几何体

#     for edge_idx in short_edge_indices:
#         p1, p2 = edges[edge_idx]
#         edge_vec = p2 - p1
#         edge_length = np.linalg.norm(edge_vec)
#         if edge_length < 1e-6:
#             continue
#         edge_unit = edge_vec / edge_length

#         # 缩短边以避免角点噪声影响
#         if edge_length > 2 * shorten_dist:
#             # 边足够长时，从两端各缩短指定距离
#             shortened_p1 = p1 + shorten_dist * edge_unit
#             shortened_p2 = p2 - shorten_dist * edge_unit
#         else:
#             # 边较短时，在中点附近创建小范围
#             mid_point = (p1 + p2) / 2
#             half_range = edge_length / 4
#             shortened_p1 = mid_point - half_range * edge_unit
#             shortened_p2 = mid_point + half_range * edge_unit

#         # print(f"短边 {edge_idx}: 原始长度 {edge_length:.3f}m, 缩短后长度 {np.linalg.norm(shortened_p2 - shortened_p1):.3f}m")

#         # 计算垂直方向（指向矩形外部）
#         perp_vec = np.array([-edge_unit[1], edge_unit[0]])
#         rect_center = np.mean(refined_corners_yz, axis=0)
#         edge_center = (p1 + p2) / 2
#         to_center = rect_center - edge_center
#         if np.dot(perp_vec, to_center) > 0:
#             perp_vec = -perp_vec

#         # 为缩短边的两个端点创建搜索区域
#         for end_idx, end_point in enumerate([shortened_p1, shortened_p2]):
#             region_info = {
#                 'center_yz': end_point,
#                 'edge_idx': edge_idx,
#                 'end_idx': end_idx,
#                 'perp_direction': perp_vec,
#                 'edge_direction': edge_unit,
#                 'points': []
#             }

#             # 使用KD-tree进行高效的区域查询
#             search_radius = max(search_rect_width, z_expand_max - z_expand_min)
#             candidate_indices = kdtree_yz.query_radius([end_point], r=search_radius)[0]
            
#             # 对候选点进行精确的几何筛选
#             for idx in candidate_indices:
#                 point = full_original_points_with_intensity[idx]
#                 point_yz = point[1:3]
#                 point_x = point[0]

#                 # 计算点相对于搜索区域中心的投影
#                 to_point = point_yz - end_point
#                 proj_along_edge = np.dot(to_point, edge_unit)   # 沿边方向投影
#                 proj_perp_edge = np.dot(to_point, perp_vec)    # 垂直方向投影

#                 # 筛选条件：在搜索矩形内 + X坐标范围内
#                 if (abs(proj_along_edge) <= search_rect_width / 2 and
#                     z_expand_min <= proj_perp_edge <= z_expand_max):
#                     if abs(point_x - avg_x_baseline) <= x_search_range:
#                         region_info['points'].append(point)

#             search_regions.append(region_info)
#             # print(f"搜索区域 {len(search_regions)-1}: KD-tree查询到 {len(candidate_indices)} 个候选点，筛选后得到 {len(region_info['points'])} 个有效点")

#             # ==================== 可视化搜索范围 ====================
#             # 创建3D可视化几何体以便调试和验证
#             if visualize_ranges:
#                 # 创建搜索区域的3D包围盒
#                 box_x_dim = x_search_range * 2
#                 box_y_dim = search_rect_width
#                 box_z_dim = (z_expand_max - z_expand_min)

#                 box = o3d.geometry.TriangleMesh.create_box(
#                     width=box_x_dim,
#                     height=box_y_dim,
#                     depth=box_z_dim
#                 )

#                 # 将盒子中心移到原点
#                 box.translate([-box_x_dim/2, -box_y_dim/2, -box_z_dim/2])

#                 # 创建旋转矩阵以对齐搜索方向
#                 R = np.identity(3)
#                 R[1, 1] = edge_unit[0]  # Y轴对齐边方向
#                 R[2, 1] = edge_unit[1]
#                 R[1, 2] = perp_vec[0]   # Z轴对齐垂直方向
#                 R[2, 2] = perp_vec[1]

#                 box.rotate(R, center=(0,0,0))

#                 # 移动到最终位置
#                 box_yz_center = end_point + perp_vec * (z_expand_min + z_expand_max) / 2
#                 final_box_center_3d = np.array([avg_x_baseline, box_yz_center[0], box_yz_center[1]])

#                 box.translate(final_box_center_3d)
#                 box.paint_uniform_color([0.0, 0.8, 0.8])  # 青色
#                 search_geometries.append(box)

#                 # 创建搜索方向指示箭头
#                 arrow_start_yz = end_point + perp_vec * z_expand_min
#                 arrow_end_yz = end_point + perp_vec * z_expand_max

#                 arrow_start_3d = np.array([avg_x_baseline, arrow_start_yz[0], arrow_start_yz[1]])
#                 arrow_end_3d = np.array([avg_x_baseline, arrow_end_yz[0], arrow_end_yz[1]])

#                 line_points = [arrow_start_3d, arrow_end_3d]
#                 line_indices = [[0, 1]]
#                 line_set = o3d.geometry.LineSet(
#                     points=o3d.utility.Vector3dVector(line_points),
#                     lines=o3d.utility.Vector2iVector(line_indices)
#                 )
#                 line_set.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
#                 search_geometries.append(line_set)

#     # print(f"创建了 {len(search_regions)} 个搜寻区域")

#     # ==================== 步骤7: 计算区域X坐标统计值 ====================
#     # 使用改进的统计策略计算每个搜索区域的最优X坐标
#     region_x_values = []
#     for i, region in enumerate(search_regions):
#         points_in_region = np.array(region['points'])
#         if len(points_in_region) > 20:
#             # 点数充足时使用统计筛选策略
#             x_coords = points_in_region[:, 0]
#             sorted_x = np.sort(x_coords)

#             if len(sorted_x) > 8:
#                 # 去除最小的8个点（可能的噪声点）
#                 filtered_x = sorted_x[8:]
#                 if len(filtered_x) >= 7:
#                     # 使用接下来最小的7个点计算平均值
#                     selected_x = filtered_x[:7]
#                     avg_x = np.mean(selected_x)
#                     region_x_values.append(avg_x)
#                     # print(f"区域 {i}: 找到 {len(points_in_region)} 个点，筛选后取最小7点平均x坐标: {avg_x:.3f}")
#                 else:
#                     # 筛选后点数不足时使用全部筛选点
#                     avg_x = np.mean(filtered_x)
#                     region_x_values.append(avg_x)
#                     # print(f"区域 {i}: 找到 {len(points_in_region)} 个点，使用 {len(filtered_x)} 个点平均x坐标: {avg_x:.3f}")
#             else:
#                 # 总点数不足时使用全部点
#                 avg_x = np.mean(sorted_x)
#                 region_x_values.append(avg_x)
#                 # print(f"区域 {i}: 找到 {len(points_in_region)} 个点，使用所有点平均X坐标: {avg_x:.3f}")
#         elif len(points_in_region) > 0:
#             # 点数较少时的处理策略
#             avg_x = np.mean(points_in_region[:, 0])
#             region_x_values.append(avg_x)
#             # print(f"区域 {i}: 只找到 {len(points_in_region)} 个点，使用平均X坐标: {avg_x:.3f}")
#         else:
#             # 未找到点时先标记为None，稍后处理
#             region_x_values.append(999)
#             # print(f"区域 {i}: 未找到点，稍后处理")
    
#     # # ==================== 步骤7.5: 处理未找到点的区域 ====================
#     # # 在所有区域搜寻完成后，单独处理没有找到点的区域
#     # for i, x_value in enumerate(region_x_values):
#     #     if x_value is None:
#     #         current_region = search_regions[i]
#     #         edge_idx = current_region['edge_idx']
#     #         end_idx = current_region['end_idx']
            
#     #         # 寻找同一条边上的另一个搜索区域
#     #         same_edge_x = None
#     #         for j, other_region in enumerate(search_regions):
#     #             if j != i and other_region['edge_idx'] == edge_idx and other_region['end_idx'] != end_idx:
#     #                 # 找到同一条边上的另一个区域，检查是否有有效的X坐标
#     #                 if region_x_values[j] is not None:
#     #                     same_edge_x = region_x_values[j]
#     #                     break
            
#     #         if same_edge_x is not None:
#     #             region_x_values[i] = 999
#     #             # print(f"区域 {i}: 使用同一条边上另一个顶点的X坐标: {same_edge_x:.3f}")
               
#     #         else:
#     #             # 如果同一条边上的另一个顶点也没有找到点，则使用全局基准值作为后备
#     #             region_x_values[i] = 999
#     #             # print(f"区域 {i}: 同边顶点也无有效坐标，使用全局平均X坐标: {avg_x_baseline:.3f}")


     
#     refined_corners_3d = np.zeros((4, 3))
 
#     # ==================== 步骤8: X坐标分配给矩形顶点 ====================
#     # 将计算出的X坐标分配给对应的矩形角点，完成3D重建
  
#     for vertex_idx in range(4):
#         vertex_yz = refined_corners_yz[vertex_idx]

#         # 计算顶点到各搜索区域中心的距离
#         distances = []
#         for region in search_regions:
#             dist = np.linalg.norm(vertex_yz - region['center_yz'])
#             distances.append(dist)

#         if len(distances) > 0:
#             # 分配最近搜索区域的X坐标
#             nearest_region_idx = np.argmin(distances)
#             assigned_x = region_x_values[nearest_region_idx]
#             refined_corners_3d[vertex_idx] = [assigned_x, vertex_yz[0], vertex_yz[1]]
#             # print(f"顶点 {vertex_idx}: 分配到区域 {nearest_region_idx}，X坐标: {assigned_x:.3f}")
#         else:
#             # 后备方案：使用全局基准X坐标
#             avg_x = avg_x_baseline
#             refined_corners_3d[vertex_idx] = [avg_x, vertex_yz[0], vertex_yz[1]]
#             # print(f"顶点 {vertex_idx}: 使用后备X坐标: {avg_x:.3f}")
# #处理点的高度差值大于5的情况，如果z轴方向同一条边上的两个顶点高度差大于5米的话，就修改高度大的点的高度为小的点的高度
#     for i in [0, 2]:
#         if np.abs(refined_corners_3d[i,0]-refined_corners_3d[(i+1)%4,0])>5:
#             diff=np.abs(refined_corners_3d[(i+2)%4,0]-refined_corners_3d[(i+3)%4,0])
#             if refined_corners_3d[i,0]>refined_corners_3d[(i+1)%4,0]:
#               if refined_corners_3d[(i+3)%4,0]>refined_corners_3d[(i+2)%4,0]:
#                 refined_corners_3d[i,0]=refined_corners_3d[(i+1)%4,0]+diff
              
#               else:
#                 refined_corners_3d[i,0]=refined_corners_3d[(i+1)%4,0]-diff
#             else:
#               if refined_corners_3d[(i+3)%4,0]>refined_corners_3d[(i+2)%4,0]:
#                 refined_corners_3d[(i+1)%4,0]=refined_corners_3d[i,0]-diff
#               else:
#                 refined_corners_3d[(i+1)%4,0]=refined_corners_3d[i,0]+diff
#     # print("--- 高精度X坐标精确化完成（增强KD-tree版本）---")
   
#     return refined_corners_3d, search_geometries,edge_sample_geometries




#采用中间的点来恢复x高度

def refine_x_coordinates_by_advanced_search(full_original_points_with_intensity, points_for_yz_refinement, refined_corners_yz, visualize_ranges=True):
    """
    通过在矩形短边周围的特定区域搜寻原始点云，使用高精度方法来精确化矩形顶点的X坐标。
    
    功能概述：
    1. 构建KD-tree索引以优化点云查询性能
    2. 计算矩形各边长度，识别短边（船舱的宽度方向）
    3. 在短边端点附近定义精细搜索区域
    4. 使用统计方法计算每个区域的最优X坐标
    5. 将计算结果分配给对应的矩形顶点
    
    
    参数:
        full_original_points_with_intensity: 完整的原始点云数据（包含强度信息）
        points_for_yz_refinement: 用于YZ平面精炼的点云数据
        refined_corners_yz: 已精炼的矩形角点YZ坐标
        visualize_ranges: 是否可视化搜索范围
    
    返回:
        refined_corners_3d: 精炼后的3D矩形角点坐标
        search_geometries: 搜索区域的可视化几何体
        []: 空列表（保持接口兼容性）
    """
  
    # print("\n--- 开始高精度X坐标精确化（增强KD-tree版本）---")
    
    # ==================== 步骤1: 构建KD-tree索引 ====================
    # 构建2D KD-tree用于YZ平面的高效空间查询
    # KD-tree将O(n)的线性搜索优化为O(log n)的树搜索
    # print("构建2D KD-tree索引（YZ平面）...")
    points_yz = full_original_points_with_intensity[:, 1:3]  # 提取YZ坐标
    kdtree_yz = KDTree(points_yz)
    # print(f"2D KD-tree构建完成，包含 {len(points_yz)} 个点")

    # ==================== 步骤2: 计算边长度和存储边信息 ====================
    # 计算矩形四条边的长度，为后续短边识别做准备
    edge_lengths = []
    edges = []
    for i in range(4):
        p1 = refined_corners_yz[i]
        p2 = refined_corners_yz[(i + 1) % 4]  # 循环连接形成闭合矩形
        edge_length = np.linalg.norm(p2 - p1)  # 计算欧几里得距离
        edge_lengths.append(edge_length)
        edges.append((p1, p2))

    # print(f"矩形四条边长度: {[f'{l:.3f}' for l in edge_lengths]}")

    # ==================== 步骤3: 计算基准X坐标 ====================
    # 使用统计方法计算基准X坐标，避免极值点的影响
    # 选择中间20%的点（60%-80%分位数）来计算更稳定的基准值
    if len(points_for_yz_refinement) > 0:
        all_x_values = points_for_yz_refinement[:, 0]
        sorted_x_values = np.sort(all_x_values)
        
        total_points = len(sorted_x_values)
        start_idx = int(total_points * 0.6)  # 60%分位数
        end_idx = int(total_points * 0.8)    # 80%分位数
        
        if start_idx < end_idx:
            # 使用中间20%的点计算基准值，排除极值影响
            middle_x_values = sorted_x_values[start_idx:end_idx]
            avg_x_baseline = np.mean(middle_x_values)
            # print(f"使用过滤后点云中X坐标中间20%的点计算基准x: {avg_x_baseline:.3f} 米")
        else:
            # 点数太少时使用全部点
            avg_x_baseline = np.mean(all_x_values)
            # print(f"使用所有过滤后点计算基准x: {avg_x_baseline:.3f} 米")
    else:
        avg_x_baseline = 25
        # print("警告: 没有可用的过滤后点来计算基准x，使用默认值 25")

    # ==================== 步骤4: 对每条边进行X坐标采样 ====================
    # 使用KD-tree优化的边缘采样，为短边识别提供数据支持
    edge_x_averages = []
    
    # 搜索参数定义
    inward_search_distance = 0.5   # 向矩形内部的搜索距离（米）
    outward_search_distance = 1.5  # 向矩形外部的搜索距离（米）
    x_sample_range = 5             # X坐标采样范围（米）
    edge_sample_geometries = []
    for i, (p1, p2) in enumerate(edges):
        # 计算边的方向向量和垂直向量
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        if edge_length < 1e-6:  # 处理退化边的情况
            edge_x_averages.append(avg_x_baseline)
            logger.debug(f"边 {i}: 长度接近零，使用全局平均值: {avg_x_baseline:.3f}")
            continue

        edge_unit = edge_vec / edge_length  # 边的单位方向向量
        perp_vec = np.array([-edge_unit[1], edge_unit[0]])  # 垂直向量（逆时针90度）
        
        # 确定垂直向量指向矩形外部
        rect_center = np.mean(refined_corners_yz, axis=0)
        edge_center = (p1 + p2) / 2
        to_center = rect_center - edge_center
        if np.dot(perp_vec, to_center) > 0:
            perp_vec = -perp_vec  # 翻转方向使其指向外部

        # 使用KD-tree进行高效的范围查询
        search_radius = max(outward_search_distance + inward_search_distance, edge_length / 2)
        candidate_indices = kdtree_yz.query_radius([edge_center], r=search_radius)[0]
        
        # 对候选点进行精确的几何筛选
        edge_sample_points = []
        for idx in candidate_indices:
            point = full_original_points_with_intensity[idx]
            point_yz = point[1:3]
            point_x = point[0]

            # 计算点到边起点的投影
            to_point = point_yz - p1
            proj_along_edge = np.dot(to_point, edge_unit)    # 沿边方向的投影
            proj_perp_edge = np.dot(to_point, perp_vec)     # 垂直边方向的投影

            # 多重条件筛选：边界内 + 搜索带内 + X坐标范围内
            if (0 <= proj_along_edge <= edge_length and
                -inward_search_distance <= proj_perp_edge <= outward_search_distance and
                avg_x_baseline - x_sample_range <= point_x <= avg_x_baseline + x_sample_range):
                edge_sample_points.append(point_x)

        # 创建边采样范围的可视化几何体
        if visualize_ranges:
            # 创建边采样区域的长方体
            box_x_dim = x_sample_range * 2
            box_y_dim = edge_length
            box_z_dim = outward_search_distance + inward_search_distance
            
            edge_box = o3d.geometry.TriangleMesh.create_box(
                width=box_x_dim,
                height=box_y_dim, 
                depth=box_z_dim
            )
            
            # 移动到原点
            edge_box.translate([-box_x_dim/2, -box_y_dim/2, -box_z_dim/2])
            
            # 创建旋转矩阵，使长方体沿着边的方向
            R = np.identity(3)
            # Y轴沿着边的方向
            R[1, 1] = edge_unit[0]  # edge_unit的Y分量
            R[2, 1] = edge_unit[1]  # edge_unit的Z分量
            # Z轴沿着垂直方向
            R[1, 2] = perp_vec[0]   # perp_vec的Y分量
            R[2, 2] = perp_vec[1]   # perp_vec的Z分量
            
            edge_box.rotate(R, center=(0,0,0))
            
            # 移动到边的中心位置
            box_yz_center = edge_center + perp_vec * (-inward_search_distance+outward_search_distance) / 2
            final_box_center_3d = np.array([avg_x_baseline, box_yz_center[0], box_yz_center[1]])
            
            edge_box.translate(final_box_center_3d)
            
            # 为不同的边设置不同的颜色
            colors = [[1.0, 0.0, 0.0],  # 红色
                     [0.0, 1.0, 0.0],  # 绿色
                     [0.0, 0.0, 1.0],  # 蓝色
                     [1.0, 1.0, 0.0]]  # 黄色
            edge_box.paint_uniform_color(colors[i])
            edge_sample_geometries.append(edge_box)
            
            # 创建边的中心线
            line_start_3d = np.array([avg_x_baseline, p1[0], p1[1]])
            line_end_3d = np.array([avg_x_baseline, p2[0], p2[1]])
            
            line_points = [line_start_3d, line_end_3d]
            line_indices = [[0, 1]]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(line_indices)
            )
            line_set.paint_uniform_color(colors[i])
            edge_sample_geometries.append(line_set)
        # 统计采样结果并计算边的X坐标平均值
        if edge_sample_points:
            sorted_points = np.sort(edge_sample_points)
            if len(sorted_points) >= 100:
                # 优先使用最小的50个点（更接近船舱边缘）
                selected_points = sorted_points[:50]
                edge_x_avg = np.mean(selected_points)
                # print(f"边 {i}: KD-tree查询到 {len(candidate_indices)} 个候选点，采样到 {len(edge_sample_points)} 个点，使用最小50个点平均x坐标: {edge_x_avg:.3f}")
            else:
                # 点数不足时使用全部点
                edge_x_avg = np.mean(sorted_points)

        else:

            edge_x_avg = 999
        edge_x_averages.append(edge_x_avg)

    # ==================== 步骤5: 识别短边 ====================
    # X坐标值较大，舱盖就不在这条边上
    sorted_indices = np.argsort(edge_x_averages)[::-1]  # 按X坐标降序排列
    short_edge_indices = sorted_indices[:2]  # 选择X坐标最大的两条边

    # 验证选择的边是否为相对边（矩形的对边）
    valid_combinations = [{0, 2}, {1, 3}]  # 有效的对边组合
    selected_set = set(short_edge_indices)

    if selected_set not in valid_combinations:
        # 如果不是对边，排除最小x坐标边及其对边，选择另一对边
        min_x_edge = sorted_indices[-1]  # x坐标最小的边
        min_x_opposite_edge = (min_x_edge + 2) % 4  # 最小x坐标边的对边
        excluded_edges = {min_x_edge, min_x_opposite_edge}
        
        # 选择剩余的另一对相对边
        all_edges = {0, 1, 2, 3}
        remaining_edges = all_edges - excluded_edges
        short_edge_indices = np.array(list(remaining_edges))
        # print(f"选出的边不是相对边，排除最小x坐标边 {min_x_edge} 及其对边 {min_x_opposite_edge}，选择另一对边: {short_edge_indices}")

    # print(f"各边x坐标平均值: {[f'{avg:.3f}' for avg in edge_x_averages]}")
    # print(f"最终选择的短边索引: {short_edge_indices}")

    # ==================== 步骤6: 定义精细搜索区域 ====================
    # 只对短边在中间范围内创建搜索区域，用于高精度X坐标计算
    
    # 搜索区域参数
    shorten_dist = 2        # 边缩短距离（米），避免角点处的噪声和角点处没有点的情况
    z_expand_min = -0.4       # Z方向向内扩展距离（米）
    z_expand_max = 0.8        # Z方向向外扩展距离（米）
    x_search_range = 5        # X方向搜索范围（米）

    search_regions = []       # 存储搜索区域信息
    search_geometries = []    # 存储可视化几何体
    edge_x_mapping = {}       # 存储每条边对应的X坐标值

    # 只对短边进行中间范围搜索
    for edge_idx in short_edge_indices:
        p1, p2 = edges[edge_idx]
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        if edge_length < 1e-6:
            continue
        edge_unit = edge_vec / edge_length

        # 缩短边以避免角点噪声影响
        if edge_length > 2 * shorten_dist:
            # 边足够长时，从两端各缩短指定距离
            shortened_p1 = p1 + shorten_dist * edge_unit
            shortened_p2 = p2 - shorten_dist * edge_unit
        else:
            # 边较短时，在中点附近创建小范围
            mid_point = (p1 + p2) / 2
            half_range = edge_length / 4
            shortened_p1 = mid_point - half_range * edge_unit
            shortened_p2 = mid_point + half_range * edge_unit

        # print(f"短边 {edge_idx}: 原始长度 {edge_length:.3f}m, 缩短后长度 {np.linalg.norm(shortened_p2 - shortened_p1):.3f}m")

        # 计算垂直方向（指向矩形外部）
        perp_vec = np.array([-edge_unit[1], edge_unit[0]])
        rect_center = np.mean(refined_corners_yz, axis=0)
        edge_center = (p1 + p2) / 2
        to_center = rect_center - edge_center
        if np.dot(perp_vec, to_center) > 0:
            perp_vec = -perp_vec

        # 在缩短边的中间范围创建搜索区域（而不是在端点）
        shortened_edge_length = np.linalg.norm(shortened_p2 - shortened_p1)
        shortened_edge_center = (shortened_p1 + shortened_p2) / 2
        
        # 创建单个搜索区域覆盖整个缩短边的中间范围
        region_info = {
            'center_yz': shortened_edge_center,
            'edge_idx': edge_idx,
            'edge_start': shortened_p1,
            'edge_end': shortened_p2,
            'edge_length': shortened_edge_length,
            'perp_direction': perp_vec,
            'edge_direction': edge_unit,
            'points': []
        }

        # 使用KD-tree进行高效的区域查询
        search_radius =  shortened_edge_length / 2
        candidate_indices = kdtree_yz.query_radius([shortened_edge_center], r=search_radius)[0]
    
        # 对候选点进行精确的几何筛选
        for idx in candidate_indices:
            point = full_original_points_with_intensity[idx]
            point_yz = point[1:3]
            point_x = point[0]

            # 计算点相对于搜索区域中心的投影
            to_point = point_yz - shortened_edge_center
            proj_along_edge = np.dot(to_point, edge_unit)   # 沿边方向投影
            proj_perp_edge = np.dot(to_point, perp_vec)    # 垂直方向投影

            # 筛选条件：在搜索矩形内 + X坐标范围内 + 在缩短边的范围内
            if (abs(proj_along_edge) <= shortened_edge_length / 2 and
                z_expand_min <= proj_perp_edge <= z_expand_max):
                if abs(point_x - avg_x_baseline) <= x_search_range:
                    region_info['points'].append(point)

        search_regions.append(region_info)
        # print(f"短边 {edge_idx} 搜索区域: KD-tree查询到 {len(candidate_indices)} 个候选点，筛选后得到 {len(region_info['points'])} 个有效点")

        # ==================== 可视化搜索范围 ====================
        # 创建3D可视化几何体以便调试和验证
        if visualize_ranges:
            # 创建搜索区域的3D包围盒
            box_x_dim = x_search_range * 2
            box_y_dim = shortened_edge_length  # 使用缩短边的长度作为Y方向尺寸
            box_z_dim = (z_expand_max - z_expand_min)

            box = o3d.geometry.TriangleMesh.create_box(
                width=box_x_dim,
                height=box_y_dim,
                depth=box_z_dim
            )

            # 将盒子中心移到原点
            box.translate([-box_x_dim/2, -box_y_dim/2, -box_z_dim/2])

            # 创建旋转矩阵以对齐搜索方向
            R = np.identity(3)
            R[1, 1] = edge_unit[0]  # Y轴对齐边方向
            R[2, 1] = edge_unit[1]
            R[1, 2] = perp_vec[0]   # Z轴对齐垂直方向
            R[2, 2] = perp_vec[1]

            box.rotate(R, center=(0,0,0))

            # 移动到最终位置
            box_yz_center = shortened_edge_center + perp_vec * (z_expand_min + z_expand_max) / 2
            final_box_center_3d = np.array([avg_x_baseline, box_yz_center[0], box_yz_center[1]])

            box.translate(final_box_center_3d)
            box.paint_uniform_color([0.0, 0.8, 0.8])  # 青色
            search_geometries.append(box)

            # 创建搜索方向指示箭头
            arrow_start_yz = shortened_edge_center + perp_vec * z_expand_min
            arrow_end_yz = shortened_edge_center + perp_vec * z_expand_max

            arrow_start_3d = np.array([avg_x_baseline, arrow_start_yz[0], arrow_start_yz[1]])
            arrow_end_3d = np.array([avg_x_baseline, arrow_end_yz[0], arrow_end_yz[1]])

            line_points = [arrow_start_3d, arrow_end_3d]
            line_indices = [[0, 1]]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(line_indices)
            )
            line_set.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            search_geometries.append(line_set)

            # 添加边界线可视化，显示缩短边的范围
            edge_line_points = [
                np.array([avg_x_baseline, shortened_p1[0], shortened_p1[1]]),
                np.array([avg_x_baseline, shortened_p2[0], shortened_p2[1]])
            ]
            edge_line_indices = [[0, 1]]
            edge_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(edge_line_points),
                lines=o3d.utility.Vector2iVector(edge_line_indices)
            )
            edge_line_set.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
            search_geometries.append(edge_line_set)

    # print(f"创建了 {len(search_regions)} 个短边搜寻区域")

    # ==================== 步骤7: 计算短边区域X坐标统计值 ====================
    # 使用改进的统计策略计算每个短边搜索区域的最优X坐标
    for i, region in enumerate(search_regions):
        points_in_region = np.array(region['points'])
        edge_idx = region['edge_idx']
        
        if len(points_in_region) > 120:
            # 点数充足时使用统计筛选策略
            x_coords = points_in_region[:, 0]
            sorted_x = np.sort(x_coords)

            if len(sorted_x) > 50:
                # 去除最小的50个点（可能的噪声点）
                filtered_x = sorted_x[50:]
                if len(filtered_x) >= 50:
                    # 使用接下来最小的50个点计算平均值
                    selected_x = filtered_x[:50]
                    avg_x = np.mean(selected_x)
                    edge_x_mapping[edge_idx] = avg_x
                    # print(f"短边 {edge_idx}: 找到 {len(points_in_region)} 个点，筛选后取最小50点平均x坐标: {avg_x:.3f}")
                else:
                    # 筛选后点数不足时使用全部筛选点
                    avg_x = np.mean(filtered_x)
                    edge_x_mapping[edge_idx] = avg_x
                    # print(f"短边 {edge_idx}: 找到 {len(points_in_region)} 个点，使用 {len(filtered_x)} 个点平均x坐标: {avg_x:.3f}")
            else:
                # 总点数不足时使用全部点
                avg_x = np.mean(sorted_x)
                edge_x_mapping[edge_idx] = avg_x
                # print(f"短边 {edge_idx}: 找到 {len(points_in_region)} 个点，使用所有点平均X坐标: {avg_x:.3f}")
        elif len(points_in_region) > 0:
            # 点数较少时的处理策略
            avg_x = np.mean(points_in_region[:, 0])
            edge_x_mapping[edge_idx] = avg_x

        else:

            pass

    # ==================== 步骤8: X坐标分配给矩形顶点 ====================
    # 将计算出的X坐标分配给对应的矩形角点，完成3D重建
    refined_corners_3d = np.zeros((4, 3))
    
    for vertex_idx in range(4):
        vertex_yz = refined_corners_yz[vertex_idx]
        
        # 确定与该顶点相关的边
        # 顶点i连接到边i和边(i-1)%4
        connected_edges = [vertex_idx, (vertex_idx - 1) % 4]
        
        # 寻找与该顶点相关的边中有有效X坐标的短边
        assigned_x = None
        for edge_idx in connected_edges:
            if edge_idx in edge_x_mapping:
                assigned_x = edge_x_mapping[edge_idx]
                # print(f"顶点 {vertex_idx}: 使用短边 {edge_idx} 的X坐标: {assigned_x:.3f}")
                break
        
        if assigned_x is None:
            # 如果相关的边都不是短边或没有有效的X坐标，使用全局基准X坐标
            assigned_x = avg_x_baseline
            # print(f"顶点 {vertex_idx}: 相关边无短边坐标，使用全局平均X坐标: {assigned_x:.3f}")
        
        refined_corners_3d[vertex_idx] = [assigned_x, vertex_yz[0], vertex_yz[1]]


   
    return refined_corners_3d, search_geometries,edge_sample_geometries




# --- 最终3D结果可视化函数 ---
def visualize_final_result_with_search_ranges(original_points, hatch_corners, search_geometries=None, edge_sample_geometries=None):
    """
    显示最终的3D检测结果，包括原始点云、检测到的矩形、X坐标搜寻范围，并标注坐标轴和短边。

    Args:
        original_points (np.array): 原始点云的Open3D对象。
        hatch_corners (np.array): 精确化后的船舱口矩形3D顶点坐标，形状为 (4, 3)。
        search_geometries (list, optional): Open3D几何体列表，用于可视化X坐标搜索区域。
        edge_sample_geometries (list, optional): 空列表，用于兼容性。
    """
    # print("\n正在显示最终的3D检测结果和搜寻范围...")

    # 绘制矩形边框（红色）
    pcd_original_o3d = o3d.geometry.PointCloud()
    pcd_original_o3d.points = o3d.utility.Vector3dVector(original_points[:, :3])
    #初始化点云颜色
    x_coords_full = original_points[:, 0]
    if np.max(x_coords_full) - np.min(x_coords_full) > 1e-6:
        norm_x_full = (x_coords_full - np.min(x_coords_full)) / (np.max(x_coords_full) - np.min(x_coords_full))
    else:
        norm_x_full = np.zeros_like(x_coords_full)
    pcd_original_o3d.colors = o3d.utility.Vector3dVector(cm.get_cmap('viridis')(norm_x_full)[:,:3])
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]] # 定义连接顶点的线段索引
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(hatch_corners),
                                  lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color([1, 0, 0]) # 红色

    # 绘制矩形角点（绿色球体）
    corner_spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.1).translate(c) for c in hatch_corners]
    for s in corner_spheres:
        s.paint_uniform_color([0, 1, 0]) # 绿色

    # 识别并标注短边（黄色）
    edge_lengths = []
    edges = []
    for i in range(4):
        p1 = hatch_corners[i]
        p2 = hatch_corners[(i + 1) % 4]
        edge_length = np.linalg.norm(p2 - p1)
        edge_lengths.append(edge_length)
        edges.append((p1, p2))

    sorted_indices = np.argsort(edge_lengths) # 按边长升序排列
    short_edge_indices = sorted_indices[:2] # 选取最短的两条边

    # print(f"矩形四条边长度: {[f'{l:.3f}' for l in edge_lengths]}")
    # print(f"短边索引: {short_edge_indices} (边长: {[f'{edge_lengths[i]:.3f}' for i in short_edge_indices]})")

    short_edge_geometries = [] # 用于存储短边的可视化几何体
    for i, edge_idx in enumerate(short_edge_indices):
        p1, p2 = edges[edge_idx]

        # 绘制短边线条
        short_edge_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([p1, p2]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        short_edge_line.paint_uniform_color([1.0, 1.0, 0.0]) # 黄色
        short_edge_geometries.append(short_edge_line)

        # 在短边中点绘制小球体作为标注
        mid_point = (p1 + p2) / 2
        label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        label_sphere.translate(mid_point)
        label_sphere.paint_uniform_color([1.0, 1.0, 0.0]) # 黄色
        short_edge_geometries.append(label_sphere)

    # 绘制坐标轴 (X-红, Y-绿, Z-蓝)
    rect_center = np.mean(hatch_corners, axis=0) # 矩形中心作为坐标轴原点
    axis_length = 2.0 # 坐标轴长度

    x_axis_end = rect_center + np.array([axis_length, 0, 0])
    x_axis_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([rect_center, x_axis_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    x_axis_line.paint_uniform_color([1.0, 0.0, 0.0]) # 红色

    y_axis_end = rect_center + np.array([0, axis_length, 0])
    y_axis_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([rect_center, y_axis_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    y_axis_line.paint_uniform_color([0.0, 1.0, 0.0]) # 绿色

    z_axis_end = rect_center + np.array([0, 0, axis_length])
    z_axis_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([rect_center, z_axis_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    z_axis_line.paint_uniform_color([0.0, 0.0, 1.0]) # 蓝色

    # 坐标轴末端的小球体标签
    x_label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    x_label_sphere.translate(x_axis_end)
    x_label_sphere.paint_uniform_color([1.0, 0.0, 0.0])

    y_label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    y_label_sphere.translate(y_axis_end)
    y_label_sphere.paint_uniform_color([0.0, 1.0, 0.0])

    z_label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    z_label_sphere.translate(z_axis_end)
    z_label_sphere.paint_uniform_color([0.0, 0.0, 1.0])

    # 准备要绘制的所有几何体列表
    geometries_to_draw = [pcd_original_o3d, line_set] + corner_spheres

    geometries_to_draw.extend(short_edge_geometries) # 添加短边标注

    geometries_to_draw.extend([ # 添加坐标轴和标签
        x_axis_line, y_axis_line, z_axis_line,
        x_label_sphere, y_label_sphere, z_label_sphere
    ])

    if search_geometries: # 如果有X坐标搜索范围几何体，则添加
        geometries_to_draw.extend(search_geometries)
        # print(f"添加了 {len(search_geometries)} 个搜寻范围可视化对象")
    
    # 添加边采样几何体的可视化
    if edge_sample_geometries:
        geometries_to_draw.extend(edge_sample_geometries)
        # print(f"添加了 {len(edge_sample_geometries)} 个边采样区域可视化对象")

    # print("\n=== 坐标轴方向说明 ===")
    # print("X轴（红色）: 深度方向")
    # print("Y轴（绿色）: 水平方向")
    # print("Z轴（蓝色）: 垂直方向")
    # print("\n=== 短边标注说明 ===")
    # print("黄色线条和球体: 标注识别出的短边")
    # print(f"短边索引: {short_edge_indices}")
    # print(f"短边长度: {[f'{edge_lengths[i]:.3f}m' for i in short_edge_indices]}")

    # 使用Open3D显示所有几何体
    o3d.visualization.draw_geometries(geometries_to_draw,
                                      window_name="船舱口检测结果及X坐标搜寻范围（含短边和坐标轴标注）",
                                      width=1280, height=720)



# --- 分割煤堆 ---
def segment_coal_pile(original_points, hatch_corners_refined,eps_coal,min_samples_coal,x_changes,yz_shrink_amount,visualize_dbscan=False):

  
    """
    基于船舱口顶点进行煤堆分割
    
    Args:
        original_points (np.array): 原始点云数据，形状为 (N, 4) 包含X,Y,Z,Intensity
        hatch_corners_refined (np.array): 精炼后的船舱口四个顶点坐标，形状为 (4, 3)
    
    Returns:
        np.array: 分割出的煤堆点云，形状为 (M, 4) 包含强度信息，如果分割失败返回空数组
    """
    # 输入验证
    if hatch_corners_refined is None or len(hatch_corners_refined) == 0:
        logger.error("segment_coal_pile: 输入的船舱口顶点为空")
        return np.array([])
    
    if original_points.shape[1] < 4:
        logger.error("segment_coal_pile: 输入的原始点云数据维度不足，至少需要4维（X,Y,Z,Intensity）")
        return np.array([])
    
    # 1. 计算船舱四个顶点的x坐标平均值
    avg_x_corners = np.mean(hatch_corners_refined[:, 0])
    
    # 定义X坐标分割阈值：x坐标平均值加5米
    x_split_threshold = avg_x_corners + x_changes
    
    # 2. 将原始点云分为两部分
    # 部分一：x坐标大于阈值的点
    part_one_mask = original_points[:, 0] > x_split_threshold
    part_one = original_points[part_one_mask]
    
    # 部分二：x坐标小于等于阈值的点
    part_two_mask = original_points[:, 0] <= x_split_threshold
    part_two = original_points[part_two_mask].copy()
    
    processed_part_two = np.array([])
    
    # 3. 处理部分二
    if len(part_two) > 0:
        # 3.1 过滤掉x坐标小于顶点x坐标平均值的点
        part_two_x_filtered_min = part_two[part_two[:, 0] >= avg_x_corners]
        
        if len(part_two_x_filtered_min) > 0:
            # 3.2 计算船舱口YZ范围并收缩0.6米
            min_y_hatch = np.min(hatch_corners_refined[:, 1])
            max_y_hatch = np.max(hatch_corners_refined[:, 1])
            min_z_hatch = np.min(hatch_corners_refined[:, 2])
            max_z_hatch = np.max(hatch_corners_refined[:, 2])
            
            shrink_amount = yz_shrink_amount
            yz_min_y_shrunk = min_y_hatch + shrink_amount
            yz_max_y_shrunk = max_y_hatch - shrink_amount
            yz_min_z_shrunk = min_z_hatch + shrink_amount
            yz_max_z_shrunk = max_z_hatch - shrink_amount
            
            # 确保收缩后范围仍然有效
            if yz_min_y_shrunk >= yz_max_y_shrunk:
                center_y = (min_y_hatch + max_y_hatch) / 2
                yz_min_y_shrunk = center_y - 0.01
                yz_max_y_shrunk = center_y + 0.01
            if yz_min_z_shrunk >= yz_max_z_shrunk:
                center_z = (min_z_hatch + max_z_hatch) / 2
                yz_min_z_shrunk = center_z - 0.01
                yz_max_z_shrunk = center_z + 0.01
            
            # 过滤YZ平面范围之外的点
            yz_filtered_mask = (part_two_x_filtered_min[:, 1] >= yz_min_y_shrunk) & \
                               (part_two_x_filtered_min[:, 1] <= yz_max_y_shrunk) & \
                               (part_two_x_filtered_min[:, 2] >= yz_min_z_shrunk) & \
                               (part_two_x_filtered_min[:, 2] <= yz_max_z_shrunk)
            part_two_yz_filtered = part_two_x_filtered_min[yz_filtered_mask]
            
            # 3.3 过滤掉X坐标大于阈值的点
            processed_part_two = part_two_yz_filtered[part_two_yz_filtered[:, 0] <= x_split_threshold]
    
    # 4. 合并两部分点云
    if len(part_one) == 0 and len(processed_part_two) == 0:

        return np.array([])
    elif len(part_one) == 0:
        combined_points_for_clustering = processed_part_two
    elif len(processed_part_two) == 0:
        combined_points_for_clustering = part_one
    else:
        combined_points_for_clustering = np.vstack((part_one, processed_part_two))
    
    # 5. 对合并后的点云进行DBSCAN聚类
    if len(combined_points_for_clustering) == 0:
        logger.error("segment_coal_pile: 合并后的点云数据为空，无法进行聚类")
        return np.array([])
    
   # 对combined_points_for_clustering进行离群点滤波

  #聚类参数
    eps_coal = 1.3
    min_samples_coal = 20
    
    # 执行聚类（只使用XYZ坐标）
    labels_coal_cluster = cluster_points_dbscan(combined_points_for_clustering[:, :3], 
                                                eps=eps_coal, 
                                                min_samples=min_samples_coal,visualize_dbscan=visualize_dbscan)
    
    # 找到点数最多的聚类（排除噪声点-1）
    unique_labels_coal, counts_coal = np.unique(labels_coal_cluster[labels_coal_cluster != -1], return_counts=True)
    
    if len(unique_labels_coal) == 0:
        logger.error("segment_coal_pile: 聚类结果为空，无法找到最大聚类")
        return np.array([])
    
    # 返回最大聚类的点
    largest_cluster_label_coal = unique_labels_coal[np.argmax(counts_coal)]
    final_coal_cluster_points = combined_points_for_clustering[labels_coal_cluster == largest_cluster_label_coal]
    
    return final_coal_cluster_points


def segment_coal_pile_world_points(original_points, hatch_corners_refined,eps_coal,min_samples_coal,z_changes,xy_shrink_amount,visualize_dbscan=False):

  
    """
    基于船舱口顶点进行煤堆分割（真实坐标系版本）
    
    Args:
        original_points (np.array): 原始点云数据，形状为 (N,3) 包含X,Y,Z,Intensity（真实坐标系）
        hatch_corners_refined (np.array): 精炼后的船舱口四个顶点坐标，形状为 (4, 3)（真实坐标系）
    
    Returns:
        np.array: 分割出的煤堆点云，形状为 (M, 3) 包含强度信息，如果分割失败返回空数组
    """
    # 输入验证
    if hatch_corners_refined is None or len(hatch_corners_refined) == 0:
        logger.error("segment_coal_pile: 输入的船舱口顶点为空")
        return np.array([])
    

    
    # 1. 计算船舱四个顶点的z坐标平均值（真实坐标系中，对应雷达坐标系的x，但方向相反）
    avg_z_corners = np.mean(hatch_corners_refined[:, 2])
    
    # 定义Z坐标分割阈值：z坐标平均值减5米（因为方向相反，原来的+5变成-5）
    z_split_threshold = avg_z_corners - z_changes
    
    # 2. 将原始点云分为两部分（因为方向相反，条件也要相反）
    # 部分一：z坐标小于阈值的点（对应雷达坐标系中x坐标大于阈值）
    part_one_mask = original_points[:, 2] < z_split_threshold
    part_one = original_points[part_one_mask]
    
    # 部分二：z坐标大于等于阈值的点（对应雷达坐标系中x坐标小于等于阈值）
    part_two_mask = original_points[:, 2] >= z_split_threshold
    part_two = original_points[part_two_mask].copy()
    
    processed_part_two = np.array([])
    
    # 3. 处理部分二
    if len(part_two) > 0:
        # 3.1 过滤掉z坐标大于顶点z坐标平均值的点
        part_two_z_filtered_min = part_two[part_two[:, 2] <= avg_z_corners]
        
        if len(part_two_z_filtered_min) > 0:
            # 3.2 计算船舱口XY范围并收缩0.6米
            min_x_hatch = np.min(hatch_corners_refined[:, 0])
            max_x_hatch = np.max(hatch_corners_refined[:, 0])
            min_y_hatch = np.min(hatch_corners_refined[:, 1])
            max_y_hatch = np.max(hatch_corners_refined[:, 1])
            
            shrink_amount = xy_shrink_amount
            xy_min_x_shrunk = min_x_hatch + shrink_amount
            xy_max_x_shrunk = max_x_hatch - shrink_amount
            xy_min_y_shrunk = min_y_hatch + shrink_amount
            xy_max_y_shrunk = max_y_hatch - shrink_amount
            
            # 确保收缩后范围仍然有效
            if xy_min_x_shrunk >= xy_max_x_shrunk:
                center_x = (min_x_hatch + max_x_hatch) / 2
                xy_min_x_shrunk = center_x - 0.01
                xy_max_x_shrunk = center_x + 0.01
            if xy_min_y_shrunk >= xy_max_y_shrunk:
                center_y = (min_y_hatch + max_y_hatch) / 2
                xy_min_y_shrunk = center_y - 0.01
                xy_max_y_shrunk = center_y + 0.01
            
            # 过滤XY平面范围之外的点
            xy_filtered_mask = (part_two_z_filtered_min[:, 0] >= xy_min_x_shrunk) & \
                               (part_two_z_filtered_min[:, 0] <= xy_max_x_shrunk) & \
                               (part_two_z_filtered_min[:, 1] >= xy_min_y_shrunk) & \
                               (part_two_z_filtered_min[:, 1] <= xy_max_y_shrunk)
            part_two_xy_filtered = part_two_z_filtered_min[xy_filtered_mask]
            
            # 3.3 过滤掉Z坐标小于阈值的点
            processed_part_two = part_two_xy_filtered[part_two_xy_filtered[:, 2] >= z_split_threshold]
    
    # 4. 合并两部分点云
    if len(part_one) == 0 and len(processed_part_two) == 0:

        return np.array([])
    elif len(part_one) == 0:
        combined_points_for_clustering = processed_part_two
    elif len(processed_part_two) == 0:
        combined_points_for_clustering = part_one
    else:
        combined_points_for_clustering = np.vstack((part_one, processed_part_two))
    
    # 5. 对合并后的点云进行DBSCAN聚类
    if len(combined_points_for_clustering) == 0:
        logger.error("segment_coal_pile: 合并后的点云数据为空，无法进行聚类")
        return np.array([])
    
   # 对combined_points_for_clustering进行离群点滤波

  #聚类参数
    eps_coal = 1.3
    min_samples_coal = 20
    
    # 执行聚类（只使用XYZ坐标）
    labels_coal_cluster = cluster_points_dbscan(combined_points_for_clustering[:, :3], 
                                                eps=eps_coal, 
                                                min_samples=min_samples_coal,visualize_dbscan=visualize_dbscan)
    # 找到点数最多的聚类（排除噪声点-1）
    unique_labels_coal, counts_coal = np.unique(labels_coal_cluster[labels_coal_cluster != -1], return_counts=True)
    
    if len(unique_labels_coal) == 0:
        logger.error("segment_coal_pile: 聚类结果为空，无法找到最大聚类")
        return np.array([])
    
    # 返回最大聚类的点
    largest_cluster_label_coal = unique_labels_coal[np.argmax(counts_coal)]
    final_coal_cluster_points = combined_points_for_clustering[labels_coal_cluster == largest_cluster_label_coal]
    
    return final_coal_cluster_points



# WebSocket连接管理类
class WebSocketManager:
    def __init__(self, uri, max_reconnect_attempts=5, reconnect_delay=3):
        self.uri = uri
        self.websocket = None
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.is_connected = False
        
    async def connect(self):
        """建立WebSocket连接"""
        attempts = 0
        while self.max_reconnect_attempts == -1 or attempts < self.max_reconnect_attempts:
            try:
                print(f"尝试连接到 WebSocket 服务器: {self.uri} (第{attempts + 1}次)")
                self.websocket = await websockets.connect(self.uri, max_size=1024 * 1024 * 1024)
                self.is_connected = True
                print("WebSocket连接成功！")
                return True
            except Exception as e:
                attempts += 1
                print(f"WebSocket连接失败: {e}")
                if self.max_reconnect_attempts == -1 or attempts < self.max_reconnect_attempts:
                    print(f"等待 {self.reconnect_delay} 秒后重试...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logger.error("达到最大重连次数，连接失败")
                    return False
        return False
    
    async def disconnect(self):
        """关闭WebSocket连接"""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.close()
                print("WebSocket连接已关闭")
            except Exception as e:
                logger.error(f"关闭WebSocket连接时出错: {e}")
            finally:
                self.is_connected = False
                self.websocket = None
    
    async def ensure_connected(self):
        """确保连接处于活跃状态，如果断开则重连"""
        if not self.is_connected or self.websocket is None:
            return await self.connect()
        
        # 检查连接是否仍然活跃
        try:
            await self.websocket.ping()
            return True
        except Exception as e:
            print(f"连接检查失败: {e}，尝试重连...")
            self.is_connected = False
            return await self.connect()
    
    async def receive_message(self):
        """接收WebSocket消息"""
        if not await self.ensure_connected():
            return None
        
        try:
            message = await self.websocket.recv()
            return message
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket连接已断开")
            self.is_connected = False
            return None
        except Exception as e:
            print(f"接收消息时出错: {e}")
            return None
    
    async def send_message(self, message):
        """发送WebSocket消息"""
        if not await self.ensure_connected():
            return False
        
        try:
            await self.websocket.send(message)
            return True
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket连接已断开")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"发送消息时出错: {e}")
            return False


async def parse_point_cloud_data(data: bytes,HEADER_SIZE,POINT_SIZE,HEADER_FORMAT,POINT_FORMAT):
    """
    解析接收到的二进制点云数据。
    """
    if len(data) < HEADER_SIZE:
        print(f"数据包太小，无法解析头部。预期最小 {HEADER_SIZE} 字节，收到 {len(data)} 字节。")
        return None

    # 1. 解析头部
    try:
        header_data = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        (
                magic, version, header_len, point_size, ts_type, frame_id, is_detection_hatch,
                last_machine_position,
                current_machine_position,
                current_hatch, current_step, start_ts_raw, end_ts_raw, pkt_count, num_points,mode_flag,
                #上次抓取点的坐标
                last_capture_point_x, last_capture_point_y, last_capture_point_z,
                # 四个角点坐标 (每个角点3个double值：x, y, z)
                corner1_x, corner1_y, corner1_z,
                corner2_x, corner2_y, corner2_z,
                corner3_x, corner3_y, corner3_z,
                corner4_x, corner4_y, corner4_z,
                # 世界坐标系四个角点坐标 (每个角点3个double值：x, y, z)
                world_corner1_x, world_corner1_y, world_corner1_z,
                world_corner2_x, world_corner2_y, world_corner2_z,
                world_corner3_x, world_corner3_y, world_corner3_z,
                world_corner4_x, world_corner4_y, world_corner4_z,
                lineWidth,floorHeight,safeDistanceXNegativeInit,safeDistanceXPositiveInit,safeDistanceY0ceanInit,safeDistanceYLandInit,
                hatchDepth,retainHeight,lineGap,expansionXFront,expansionYFront,expansionXBack,expansionYBack,
                planeThreshold,heightDiff,signFlag,kValue,bValue,planeDistance,bevelDistance,blockWidth,blockLength
            ) = header_data
    except struct.error as e:
        print(f"解析头部失败: {e}")
        return None

    # 验证魔术字
    if magic != b'LPNT':
        logger.warning(f"魔术字不匹配！预期 b'LPNT'，实际 {magic}")
        return None

    print(f"\n--- 解析头部 v{version} ---")
    print(f"  Magic: {magic.decode()}")
    print(f"  Version: {version}")
    print(f"  Header Length: {header_len}")
    print(f"  Point Size: {point_size}")
    print(f"  Frame ID: {frame_id}")
    print(f"  is_detection_hatch: {is_detection_hatch}")
    print(f"  Current Hatch: {current_hatch}")
    print(f"  Current Step: {current_step}")
    print(f"  Number of Points: {num_points}")
    print(f"  Last Machine Position: {last_machine_position}")
    print(f"  Current Machine Position: {current_machine_position}")
    print(f"  Mode Flag: {mode_flag}")
    print(f"  Last Capture Point: ({last_capture_point_x:.3f}, {last_capture_point_y:.3f}, {last_capture_point_z:.3f})")
    print(f"  Corner1: ({corner1_x:.3f}, {corner1_y:.3f}, {corner1_z:.3f})")
    print(f"  Corner2: ({corner2_x:.3f}, {corner2_y:.3f}, {corner2_z:.3f})")
    print(f"  Corner3: ({corner3_x:.3f}, {corner3_y:.3f}, {corner3_z:.3f})")
    print(f"  Corner4: ({corner4_x:.3f}, {corner4_y:.3f}, {corner4_z:.3f})")
    print(f"  World Corner1: ({world_corner1_x:.3f}, {world_corner1_y:.3f}, {world_corner1_z:.3f})")
    print(f"  World Corner2: ({world_corner2_x:.3f}, {world_corner2_y:.3f}, {world_corner2_z:.3f})")
    print(f"  World Corner3: ({world_corner3_x:.3f}, {world_corner3_y:.3f}, {world_corner3_z:.3f})")
    print(f"  World Corner4: ({world_corner4_x:.3f}, {world_corner4_y:.3f}, {world_corner4_z:.3f})")

    print(f"  lineWidth: {lineWidth}")
    print(f"  floorHeight: {floorHeight}")
    print(f"  safeDistanceXNegativeInit: {safeDistanceXNegativeInit}")
    print(f"  safeDistanceXPositiveInit: {safeDistanceXPositiveInit}")
    print(f"  safeDistanceY0ceanInit: {safeDistanceY0ceanInit}")
    print(f"  safeDistanceYLandInit: {safeDistanceYLandInit}")
    print(f"  hatchDepth: {hatchDepth}")
    print(f"  retainHeight: {retainHeight}")
    print(f"  lineGap: {lineGap}")
    print(f"  expansionXFront: {expansionXFront}")
    print(f"  expansionYFront: {expansionYFront}")
    print(f"  expansionXBack: {expansionXBack}")
    print(f"  expansionYBack: {expansionYBack}")
    print(f"  planeThreshold: {planeThreshold}")
    print(f"  heightDiff: {heightDiff}")
    print(f"  signFlag: {signFlag}")
    print(f"  kValue: {kValue}")
    print(f"  bValue: {bValue}")
    print(f"  planeDistance: {planeDistance}")
    print(f"  bevelDistance: {bevelDistance}")
    print(f"  blockWidth: {blockWidth}")
    print(f"  blockLength: {blockLength}")

    print("------------------")

    # 2. 解析点数据
    points = []
    payload_data = data[HEADER_SIZE:]

    for i in range(num_points):
        start_idx = i * POINT_SIZE
        end_idx = start_idx + POINT_SIZE

        if end_idx > len(payload_data):
            logger.debug(f"警告: 点数据不完整！只解析了 {len(points)} 个点。")
            break

        point_bytes = payload_data[start_idx:end_idx]
        try:
            x_raw, y_raw, z_raw, refl, tag = struct.unpack(POINT_FORMAT, point_bytes)
            
            # 转换为米
            x_meters = x_raw / 1000.0
            y_meters = y_raw / 1000.0
            z_meters = z_raw / 1000.0
            
            points.append({'x': x_meters, 'y': y_meters, 'z': z_meters, 'refl': refl, 'tag': tag})
        except struct.error as e:
            logger.debug(f"解析点数据失败，点索引 {i}: {e}")
            break

    # print(f"成功解析 {len(points)} 个点。")
    return {
        'header': {
            'version': version,
            'frame_id': frame_id,
            'is_detection_hatch': is_detection_hatch,
            'last_machine_position': last_machine_position,
            'current_machine_position': current_machine_position,
            'current_hatch': current_hatch,
            'current_step': current_step,
            'num_points': num_points,
            'mode_flag': mode_flag,
            #上次抓取点的坐标
            'last_capture_point': {'x': last_capture_point_x, 'y': last_capture_point_y, 'z': last_capture_point_z},
            'hatch_corners': {
                'corner1': {'x': corner1_x, 'y': corner1_y, 'z': corner1_z},
                'corner2': {'x': corner2_x, 'y': corner2_y, 'z': corner2_z},
                'corner3': {'x': corner3_x, 'y': corner3_y, 'z': corner3_z},
                'corner4': {'x': corner4_x, 'y': corner4_y, 'z': corner4_z}
            },
            'world_corners': {
            'corner1': {'x': world_corner1_x, 'y': world_corner1_y, 'z': world_corner1_z},
            'corner2': {'x': world_corner2_x, 'y': world_corner2_y, 'z': world_corner2_z},
            'corner3': {'x': world_corner3_x, 'y': world_corner3_y, 'z': world_corner3_z},
            'corner4': {'x': world_corner4_x, 'y': world_corner4_y, 'z': world_corner4_z}
        },

        'lineWidth': lineWidth,
        'floorHeight': floorHeight,
        'safeDistanceXNegativeInit': safeDistanceXNegativeInit,
        'safeDistanceXPositiveInit': safeDistanceXPositiveInit,
        'safeDistanceY0ceanInit': safeDistanceY0ceanInit,
        'safeDistanceYLandInit': safeDistanceYLandInit,
        'hatchDepth': hatchDepth,
        'retainHeight': retainHeight,
        'lineGap': lineGap,
        'expansionXFront': expansionXFront,
        'expansionYFront': expansionYFront,
        'expansionXBack': expansionXBack,
        'expansionYBack': expansionYBack,
        'planeThreshold': planeThreshold,
        'heightDiff': heightDiff,
        'signFlag': signFlag,
        'kValue': kValue,
        'bValue': bValue,
        'planeDistance': planeDistance,
        'bevelDistance': bevelDistance,
        'blockWidth': blockWidth,
        'blockLength': blockLength,
        
        },
        'points': points
        
    }




async def get_point_cloud_from_websocket_persistent(ws_manager,HEADER_SIZE,POINT_SIZE,HEADER_FORMAT,POINT_FORMAT):
    """
    从持久WebSocket连接获取点云数据并转换为numpy数组格式。
    """
    
    if not ws_manager:
            print("WebSocket管理器未初始化")
            raise websockets.exceptions.ConnectionClosed(None, None, "WebSocket管理器未初始化")
        
    # print("等待接收点云数据...")
    
    while True:
        message = await ws_manager.receive_message()
   
        if message is None:
            print("接收消息失败")
            # 检查连接状态，如果断开则抛出WebSocket异常
            if not ws_manager.is_connected:
               
                raise websockets.exceptions.WebSocketException("WebSocket连接已断开")
            else:
                # 如果连接正常但接收失败，抛出通用WebSocket异常
                raise websockets.exceptions.WebSocketException("接收WebSocket消息失败")
        # 检查是否为二进制数据
        if isinstance(message, bytes):
            # print(f"接收到二进制点云数据，长度: {len(message)} 字节")
            
            try:
                # 直接解析二进制点云数据
                parsed_data = await parse_point_cloud_data(message,HEADER_SIZE,POINT_SIZE,HEADER_FORMAT,POINT_FORMAT)
                if parsed_data and parsed_data['points']:
                    print(f"\n=== 成功获取点云数据 ===")
                    # print(f"点数: {len(parsed_data['points'])}")
                    
                    # 转换为numpy数组格式 (N, 4) - X, Y, Z, Intensity
                    points_list = []
                    for point in parsed_data['points']:
                        points_list.append([
                            point['x'],
                            point['y'], 
                            point['z'],
                            point['refl']  # 使用反射强度作为Intensity
                        ])
                    
                    points_array = np.array(points_list, dtype=np.float32)
                    # print(f"转换为numpy数组: {points_array.shape}")
                    return points_array, parsed_data['header']
                    
            except Exception as e:
                print(f"解析二进制数据失败: {e}")
                continue
                
        elif isinstance(message, str):
            # 保留对字符串消息的处理（用于兼容性）
            try:
                json_data = json.loads(message)
                
                # 检查是否包含点云数据
                if 'data$okio' in json_data and isinstance(json_data['data$okio'], str):
                    base64_data_str = json_data['data$okio']
                    print(f"接收到Base64编码的点云数据，长度: {len(base64_data_str)}")
                    
                    try:
                        decoded_bytes = base64.b64decode(base64_data_str)
                        print(f"Base64解码成功，得到 {len(decoded_bytes)} 字节原始数据。")
                        
                        # 解析点云数据
                        parsed_data = await parse_point_cloud_data(decoded_bytes,HEADER_SIZE,POINT_SIZE,HEADER_FORMAT,POINT_FORMAT)
                        if parsed_data and parsed_data['points']:
                            print(f"\n=== 成功获取点云数据 ===")
                            print(f"点数: {len(parsed_data['points'])}")
                            
                            # 转换为numpy数组格式 (N, 4) - X, Y, Z, Intensity
                            points_list = []
                            for point in parsed_data['points']:
                                points_list.append([
                                    point['x'],
                                    point['y'], 
                                    point['z'],
                                    point['refl']  # 使用反射强度作为Intensity
                                ])
                            
                            points_array = np.array(points_list, dtype=np.float32)
                            print(f"转换为numpy数组: {points_array.shape}")
                            return points_array, parsed_data['header']
                            
                    except base64.binascii.Error as e:
                        print(f"Base64解码失败: {e}")
                        continue
                        
            except json.JSONDecodeError:
                print("收到非JSON消息，跳过")
                continue




# 添加WebSocket服务器类
class CoalPileBroadcastServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self.clients = set()  # 存储所有连接的客户端
        self.server = None
        
    async def register_client(self, websocket, path=None):
        """注册新的客户端连接"""
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"新客户端连接: {client_info}, 当前连接数: {len(self.clients)}")
        
        try:   
            # 保持连接活跃，等待客户端断开
            await websocket.wait_closed()
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端 {client_info} 正常断开连接")
        except Exception as e:
            logger.error(f"客户端 {client_info} 连接异常: {e}")
        finally:
            # 移除断开的客户端
            self.clients.discard(websocket)
            logger.info(f"客户端 {client_info} 已移除, 当前连接数: {len(self.clients)}")
    
    async def broadcast_coal_pile_data(self, coal_pile_data):
        """广播煤堆数据到所有连接的客户端"""
        if not self.clients:
            logger.warning("没有连接的客户端，跳过广播")
            return
        
        # 准备广播消息
        broadcast_msg = {
            "type": "coal_pile_broadcast",
            "timestamp": datetime.now().isoformat(),
            "data": coal_pile_data
        }
        
        message = json.dumps(broadcast_msg, ensure_ascii=False)
        
        # 广播到所有客户端
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send(message)
                logger.debug(f"数据已发送到客户端: {client.remote_address}")
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
                logger.info(f"客户端 {client.remote_address} 连接已断开")
            except Exception as e:
                disconnected_clients.add(client)
                logger.error(f"发送数据到客户端 {client.remote_address} 失败: {e}")
        
        # 移除断开的客户端
        for client in disconnected_clients:
            self.clients.discard(client)
        
        successful_sends = len(self.clients) - len(disconnected_clients)
        logger.info(f"煤堆数据已广播到 {successful_sends} 个客户端")
    
    async def start_server(self):
        """启动WebSocket服务器"""
        logger.info(f"启动煤堆点云广播服务器: ws://{self.host}:{self.port}")
        
        # 启动WebSocket服务器
        self.server = await websockets.serve(
            self.register_client,
            self.host,
            self.port,
            max_size=1024 * 1024 * 1024,  # 1024MB最大消息大小
            ping_interval=20,  # 20秒ping间隔
            ping_timeout=10    # 10秒ping超时
        )
        
        logger.info(f"广播服务器已启动，等待客户端连接...")
        return self.server


def lidar_to_world_with_x(points_lidar, translation, actual_x, rotation_angles=[0, 0, 0], degrees=True):
    """
    将激光雷达坐标系下的点转换到实际坐标系，并加入实际x值

    参数:
        points_lidar: Nx3 numpy数组，激光雷达坐标系下的点云 (x, y, z)
        translation: 3元素列表/数组，激光雷达中心在实际坐标系中的位置 [tx, ty, tz]
        actual_x: 实际x值，将加入到平移矩阵中
        rotation_angles: 3元素列表/数组，安装角度 [roll, pitch, yaw]（默认单位：度）
        degrees: 是否为角度制（True），False则为弧度制

    返回:
        Nx3 numpy数组，实际坐标系下的点云
    """

    points_lidar = np.asarray(points_lidar)
    if points_lidar.ndim == 1:  # 兼容单个点
        points_lidar = points_lidar[np.newaxis, :]

    # 1) 轴向置换矩阵：把 L 系 (x,y,z) 映射到与 W 系轴向一致
    #    [Lz,  Ly, -Lx]  -> [Wx, Wy, Wz]
    P = np.array([[0, 0, 1],   # Wx =  Lz
                  [0, 1, 0],   # Wy =  Ly
                  [-1, 0, 0]]) # Wz = -Lx

    # 2) 安装角：支持三个轴的旋转 (roll, pitch, yaw)
    # 用户可以自己调整这三个角度
    R_install = R.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()

    # 3) 组合旋转：先轴向置换，再安装角
    R_total = R_install @ P

    # 4) 平移向量，加入实际x值
    T = np.array([translation[0] + actual_x, translation[1], translation[2]]).reshape(3, 1)

    # 5) 转换
    points_world = (R_total @ points_lidar.T) + T
    return points_world.T



def world_to_lidar_with_x(points_world, translation, rotation_angles=[0, 0, 0], degrees=True):
    """
    将实际坐标系下的点转换到激光雷达坐标系，并减去实际x值

    参数:
        points_world: Nx3 numpy数组，实际坐标系下的点云 (x, y, z)
        translation: 3元素列表/数组，激光雷达中心在实际坐标系中的位置 [tx, ty, tz]
        actual_x: 实际x值，将从平移矩阵中减去
        rotation_angles: 3元素列表/数组，安装角度 [roll, pitch, yaw]（默认单位：度）
        degrees: 是否为角度制（True），False则为弧度制

    返回:
        Nx3 numpy数组，激光雷达坐标系下的点云
    """
    
    points_world = np.asarray(points_world)
    if points_world.ndim == 1:  # 兼容单个点
        points_world = points_world[np.newaxis, :]
    
    # 1) 轴向置换矩阵：把 L 系 (x,y,z) 映射到与 W 系轴向一致
    #    [Lz,  Ly, -Lx]  -> [Wx, Wy, Wz]
    P = np.array([[0, 0, 1],   # Wx =  Lz
                  [0, 1, 0],   # Wy =  Ly
                  [-1, 0, 0]]) # Wz = -Lx
    
    # 2) 安装角：支持三个轴的旋转 (roll, pitch, yaw)
    R_install = R.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()
    
    # 3) 组合旋转：先轴向置换，再安装角
    R_total = R_install @ P
    
    # 4) 平移向量，加入实际x值
    T = np.array([translation[0], translation[1], translation[2]]).reshape(3, 1)
    
    # 5) 逆向转换：先减去平移，再应用逆旋转
    # 逆旋转矩阵是原旋转矩阵的转置
    R_total_inv = R_total.T
    
    # 减去平移向量
    points_translated = points_world.T - T
    
    # 应用逆旋转
    points_lidar = R_total_inv @ points_translated
    
    return points_lidar.T




def process_blocks_for_line(blocks_heights, world_coal_pile_points, lines_dict, 
                            y_land, y_ocean,
                             block_width, block_length,
                            enable_limited_flag, logger):
    """
    对当前线进行点云分块处理，并更新 blocks_heights 字典

    Args:
        blocks_heights: 用于存储所有分块高度的字典
        world_coal_pile_points: 世界坐标系下的煤堆点云
        lines_dict: 线段字典
        current_line: 当前线段
        y_land, y_ocean: 陆地和海洋侧Y坐标
        hatch_height: 舱口高度
        current_line_height: 当前线高度
        floor_height: 层高
        block_width, block_length: 分块尺寸
        exclude_x_center, exclude_y_center, exclude_radius: 排除区域参数
        enable_limited_flag: 是否启用边界限制
        logger: 日志记录器
    """
    for current_line in lines_dict:

        current_line_points = world_coal_pile_points[
            (world_coal_pile_points[:, 0] >= lines_dict[current_line][0]) & 
            (world_coal_pile_points[:, 0] <= lines_dict[current_line][1])
        ]
        current_line_points = current_line_points[
            (current_line_points[:, 1] <= y_ocean) & 
            (current_line_points[:, 1] >= y_land)
        ]


        x_min, x_max = lines_dict[current_line]
        y_min, y_max = y_land, y_ocean

        original_n_y = int(floor((y_max - y_min) / block_length))
        original_n_x = int(floor((x_max - x_min) / block_width))

        if enable_limited_flag:
            remaining_y = (y_max - y_min) - original_n_y * block_length
            n_y = original_n_y + (1 if remaining_y > 0.1 else 0)
            n_x = original_n_x
        else:
            n_y = original_n_y
            n_x = original_n_x

        logger.info(f"线 {current_line} 的总块数为：{n_y * n_x}")

        for i in range(n_y):
            for j in range(n_x):
                x_min_block = x_min + j * block_width
                x_max_block = x_min + (j + 1) * block_width
                y_min_block = y_min + i * block_length
                y_max_block = y_min + (i + 1) * block_length

                if enable_limited_flag and i == n_y - 1 and remaining_y > 0.1:
                    y_max_block = y_max

                current_line_points_block = current_line_points[
                    (current_line_points[:, 0] >= x_min_block) & 
                    (current_line_points[:, 0] <= x_max_block) & 
                    (current_line_points[:, 1] >= y_min_block) & 
                    (current_line_points[:, 1] <= y_max_block)
                ]

                if len(current_line_points_block) > 0:
                    current_line_points_block_height = current_line_points_block[:, 2].mean()
                else:
                    current_line_points_block_height = np.nan
                    logger.warning(f"线 {current_line} 的分块({i},{j})没有找到点云数据，设置高度为NaN")

                blocks_heights[(i, j+( -1)*n_x)] = current_line_points_block_height



def calculate_capture_point(world_coal_pile_points, lines_dict, current_line, 
                          exclude_x_center, exclude_y_center, exclude_radius,x_left,x_right,y_front,y_back,
                          y_land, y_ocean, hatch_height, current_line_height, 
                          floor_height, block_width, block_length, line_width,
                          plane_threshold, plane_distance, bevel_distance, Sign, k, b, above_current_line_layer_min_height,enable_limited_flag,limited_height,
                          x_dump_truck,y_dump_truck,limited_change_height,above_current_line_layer2_min_height,mode_flag,land_to_centerline,ocean_to_centerline,logger):
    """
    计算抓取点的核心函数
    
    Args:
        world_coal_pile_points: 世界坐标系下的煤堆点云
        lines_dict: 线段字典
        current_line: 当前线段
        exclude_x_center: 排除区域中心X坐标
        exclude_y_center: 排除区域中心Y坐标  
        exclude_radius: 排除区域半径
        y_land: 陆地侧Y坐标
        y_ocean: 海洋侧Y坐标
        hatch_height: 舱口高度
        current_line_height: 当前线高度
        floor_height: 层高
        block_width: 分块宽度
        block_length: 分块长度
        line_width: 线宽
        plane_threshold: 平面阈值
        plane_distance: 平面距离
        bevel_distance: 斜面距离
        Sign: 是否使用线性函数
        k: 线性函数的斜率
        b: 线性函数的截距
        logger: 日志记录器
        
    Returns:
        dict: 包含抓取点坐标的字典 {'x': float, 'y': float, 'z': float}
    """
    # 计算排除区域边界
    remove_x_negative = exclude_x_center - exclude_radius
    remove_x_positive = exclude_x_center + exclude_radius
    remove_y_negative = exclude_y_center - exclude_radius
    remove_y_positive = exclude_y_center + exclude_radius

    # 获取当前线的点云
    current_line_points = world_coal_pile_points[
        (world_coal_pile_points[:, 0] >= lines_dict[current_line][0]) & 
        (world_coal_pile_points[:, 0] <= lines_dict[current_line][1])
    ]
    # 提取在当前线框内且y坐标在安全范围内的点
    current_line_points = current_line_points[
        (current_line_points[:, 1] <= y_ocean) & 
        (current_line_points[:, 1] >= y_land)
    ]
    
    # 计算当前线在哪一层
    current_line_layer = math.ceil(abs(hatch_height - current_line_height) / floor_height)
    
    # 将当前线的点云分成多个分块
    current_line_points_blocks = {}
    current_line_points_blocks_heights = {}
    x_min, x_max = lines_dict[current_line]
    y_min, y_max = y_land, y_ocean

    # 计算原始的分块数量
    original_n_y = int(floor((y_max - y_min) / block_length))
    original_n_x = int(floor((x_max - x_min) / block_width))
    
    if enable_limited_flag:

        # 计算y方向的剩余距离
        remaining_y = (y_max - y_min) - original_n_y * block_length
        # 如果y方向有剩余距离，则增加一个分块
        n_y = original_n_y + (1 if remaining_y > 0.1 else 0)
        n_x = original_n_x  # x方向不变，因为刚好整除
    
    else:
        n_y = original_n_y
        n_x = original_n_x


    
    # 总共分了多少块
    total_blocks = n_y * n_x
    logger.info(f"当前线内的总块数为：{total_blocks}")

    # 分块处理
    for i in range(n_y):
        for j in range(n_x):
            # 计算当前分块的坐标
            x_min_block = x_min + j * block_width
            x_max_block = x_min + (j + 1) * block_width
            y_min_block = y_min + i * block_length
            y_max_block = y_min + (i + 1) * block_length

            if enable_limited_flag:
                if i == n_y - 1 and remaining_y > 0.1:
                     y_max_block = y_max  # 扩展到实际边界
                else:
                     y_max_block = y_min + (i + 1) * block_length



            # 计算当前分块的面积
            block_area = block_width * (y_max_block - y_min_block)
            # 提取当前分块的点云
            current_line_points_block = current_line_points[
                (current_line_points[:, 0] >= x_min_block) & 
                (current_line_points[:, 0] <= x_max_block) & 
                (current_line_points[:, 1] >= y_min_block) & 
                (current_line_points[:, 1] <= y_max_block)
            ]
            
            # 将当前分块的点云存到字典中
            current_line_points_blocks[(i, j)] = current_line_points_block
            
            # 计算当前分块的高度
            if len(current_line_points_block) > 0:
                current_line_points_block_height = current_line_points_block[:, 2].mean()
            else:
                current_line_points_block_height = np.nan
                logger.warning(f"分块({i},{j})没有找到点云数据，设置高度为NaN")
            
            # 将当前分块的高度存到字典中
            current_line_points_blocks_heights[(i, j)] = current_line_points_block_height

            
            if mode_flag!=2 and mode_flag!=3: #海陆侧抓取模式

                # 计算分块与排除区域的交集面积
                overlap_x_min = max(x_min_block, remove_x_negative)
                overlap_x_max = min(x_max_block, remove_x_positive)
                overlap_y_min = max(y_min_block, remove_y_negative)
                overlap_y_max = min(y_max_block, remove_y_positive)
                
                # 判断是否有交集
                if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
                    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
                    overlap_ratio = overlap_area / block_area
                    # 如果当前分块有30%以上的范围落入了排除区域，就将这个分块的平均高度设置为999
                    if overlap_ratio >= 0.3:
                        current_line_points_blocks_heights[(i, j)] = 999
                        logger.info(f"分块({i},{j})与排除区域重叠比例为 {overlap_ratio:.2f}，设置高度为999")
                        continue




        
    # 选取连续的面积为24的块，统计这些块的平均高度值
    window_size = 6

    if mode_flag==2 or mode_flag==3: #海陆侧抓取模式
    #将划分的这些行，分成两个区域，一个是陆侧，一个是海侧，偶数的话将这些行平均分成两半，奇数的话将这些行平均分成两半，但是多出来的那一行，作为陆侧的，然后将海陆侧的间隔坐标作为中心线
        if n_y % 2 == 0:
            land_start = 0
            separate = n_y // 2
            ocean_start=separate+(ocean_to_centerline//block_width)
            land_end=separate-(land_to_centerline//block_width)-1
            ocean_end=n_y-1
            logger.info(f"陆侧的开始({land_start})，陆侧的结束({land_end})，陆侧的块数({land_end-land_start+1})，海侧的开始({ocean_start})，海侧的结束({ocean_end})，海侧的块数({ocean_end-ocean_start+1})")
        
        else:
            land_start = 0
            separate= n_y // 2 + 1
            ocean_start=separate+(ocean_to_centerline//block_width)
            land_end=separate-(land_to_centerline//block_width)-1
            ocean_end=n_y-1
            logger.info(f"陆侧的开始({land_start})，陆侧的结束({land_end})，陆侧的块数({land_end-land_start+1})，海侧的开始({ocean_start})，海侧的结束({ocean_end})，海侧的块数({ocean_end-ocean_start+1})")
        
        if mode_flag==2: #海侧抓取模式

            if ocean_end-ocean_start+1<window_size:
                logger.info(f"海侧的块数({ocean_end-ocean_start+1})小于窗口大小({window_size})，无法进行抓取")
                logger.info(f"直接从中心线分海陆侧区域")
                ocean_start=separate
                logger.info(f"海侧的开始({ocean_start})，海侧的结束({ocean_end})，海侧的块数({ocean_end-ocean_start+1})")
                if ocean_end-ocean_start+1<window_size:
                    logger.info(f"海侧的块数({ocean_end-ocean_start+1})小于窗口大小({window_size})，无法进行抓取")
                    logger.info(f"直接从最海侧取一个抓取点")
                    ocean_start=ocean_end-window_size+1
                    logger.info(f"海侧的开始({ocean_start})，海侧的结束({ocean_end})，海侧的块数({ocean_end-ocean_start+1})")



            

            #current_line_points_blocks_heights中只保留属于海侧的块的高度，不属于海侧的其他块的高度设为999
            for i in range(n_y):
                for j in range(n_x):
                    # 检查该块是否在海侧范围内
                    if i < ocean_start or i > ocean_end:
                        # 不在海侧范围内，将高度设为999
                        current_line_points_blocks_heights[(i, j)] = 999
                        logger.info(f"分块({i},{j})不在海侧范围内，设置高度为999")

        if mode_flag==3: #陆侧抓取模式
            if land_end-land_start+1<window_size:
                logger.info(f"陆侧的块数({land_end-land_start+1})小于窗口大小({window_size})，无法进行抓取")
                logger.info(f"直接从中心线分海陆侧区域")
                land_end=separate-1
                logger.info(f"陆侧的开始({land_start})，陆侧的结束({land_end})，陆侧的块数({land_end-land_start+1})")
                if land_end-land_start+1<window_size:
                    logger.info(f"陆侧的块数({land_end-land_start+1})小于窗口大小({window_size})，无法进行抓取")
                    logger.info(f"直接从最陆侧取一个抓取点")
                    land_end=land_start+window_size-1
                    logger.info(f"陆侧的开始({land_start})，陆侧的结束({land_end})，陆侧的块数({land_end-land_start+1})")


            
            #current_line_points_blocks_heights中只保留属于陆侧的块的高度，不属于陆侧的其他块的高度设为999
            for i in range(n_y):
                for j in range(n_x):
                    # 检查该块是否在陆侧范围内
                    if i < land_start or i > land_end:
                        # 不在陆侧范围内，将高度设为999
                        current_line_points_blocks_heights[(i, j)] = 999
                        logger.info(f"分块({i},{j})不在陆侧范围内，设置高度为999")
    
            
    # 打印存储的分块高度
    for (i, j), height in current_line_points_blocks_heights.items():
        logger.info(f"分块({i},{j})的高度为：{height}")

    
    # 遍历所有可能的起点
    best_start_y = None
    best_avg_height = -np.inf
    best_blocks = None
    best_block_heights = []
    
    for start_y in range(0, n_y - window_size + 1):
        heights = []
        has_invalid_block = False  # 标记是否包含高度为999的块
        
        for i in range(start_y, start_y + window_size):
            for j in range(n_x):
                h = current_line_points_blocks_heights.get((i, j))
                if h == 999:
                    has_invalid_block = True
                    break  # 提前跳出当前列
                if h is not None and not np.isnan(h):
                    heights.append(h)
            if has_invalid_block:
                break  # 提前跳出当前行

        if has_invalid_block:
            logger.info(f"窗口起始于 y={start_y} 的块包含无效高度999，跳过")
            continue

        if heights:
            avg_height = np.mean(heights)
            if avg_height > best_avg_height:
                # 记录当前块的高度
                best_block_heights = heights
                best_avg_height = avg_height
                best_start_y = start_y
                best_blocks = [(i, j) for i in range(start_y, start_y + window_size) for j in range(n_x)]
                # 打印当前最佳块
                logger.info(f"当前最佳块: {best_blocks}，平均高度: {best_avg_height}")

    # 检查是否找到了最佳结果
    if best_blocks is None or best_avg_height == -np.inf:
        error_msg = "未找到最佳结果，所有可能的窗口都包含无效高度或没有有效数据"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    # 打印最佳块
    logger.info(f"最佳块: {best_blocks}，平均高度: {best_avg_height}")
    
    #取这些块的最高高度
    max_height = max(best_block_heights)
    logger.info(f"最佳块的最高高度: {max_height}")

    # 计算这些块的高度的方差
    height_var = np.var(best_block_heights)
    logger.info(f"{window_size*(line_width/block_width)}块的高度的方差: {height_var}")
    
    # 计算中心点坐标
    center_xs = []
    center_ys = []
    for (i, j) in best_blocks:
        # X 中心
        x_center = x_min + (j + 0.5) * block_width
        # Y 中心
        y_center = y_min + (i + 0.5) * block_length
        center_xs.append(x_center)
        center_ys.append(y_center)

    # 计算 XY 平面中心点
    avg_x = np.mean(center_xs)
    if abs(avg_x-x_left)<=x_dump_truck or abs(avg_x-x_right)<=x_dump_truck:
        x_dump_truck_flag=True
    else:
        x_dump_truck_flag=False


    avg_y = np.mean(center_ys)
    if abs(avg_y-y_front)<=y_dump_truck or abs(avg_y-y_back)<=y_dump_truck:
        y_dump_truck_flag=True
    else:
        y_dump_truck_flag=False

    if avg_y>=y_front:
        avg_y=y_front
    
    if avg_y<=y_back:
        avg_y=y_back



    
    # 判断这些块的高度的方差是否小于阈值
    if height_var < plane_threshold:
        # 如果小于阈值，就认为这些块是平面的
        logger.info(f"这{window_size*(line_width/block_width)}块是平面的")
        if Sign==1:
            if enable_limited_flag and y_dump_truck_flag and not x_dump_truck_flag:
                avg_height = max_height + k*height_var+b
                logger.info(f"高度改变了{k*height_var+b}米，改变前为{max_height}，改变后为{avg_height}")
            else:
                avg_height = best_avg_height + k*height_var+b
                logger.info(f"高度改变了{k*height_var+b}米，改变前为{best_avg_height}，改变后为{avg_height}")
        else:
            if enable_limited_flag and y_dump_truck_flag and not x_dump_truck_flag:
                avg_height = max_height + plane_distance
                logger.info(f"高度改变了{plane_distance}米，改变前为{max_height}，改变后为{avg_height}")
            else:
                avg_height = best_avg_height + plane_distance
                logger.info(f"高度改变了{plane_distance}米，改变前为{best_avg_height}，改变后为{avg_height}")
    else:
        # 如果大于阈值，就认为这些块不是平面的
        logger.info(f"这{window_size*(line_width/block_width)}块是斜面的")
        if Sign==1:
            if enable_limited_flag and y_dump_truck_flag and not x_dump_truck_flag:
                avg_height = max_height + k*height_var+b
                logger.info(f"高度改变了{k*height_var+b}米，改变前为{max_height}，改变后为{avg_height}")
            else:
                avg_height = best_avg_height + k*height_var+b
                logger.info(f"高度改变了{k*height_var+b}米，改变前为{best_avg_height}，改变后为{avg_height}")
        else:

            if enable_limited_flag and y_dump_truck_flag and not x_dump_truck_flag:
                avg_height = max_height + bevel_distance
                logger.info(f"高度改变了{bevel_distance}米，改变前为{max_height}，改变后为{avg_height}")
            else:
                avg_height = best_avg_height + bevel_distance
                logger.info(f"高度改变了{bevel_distance}米，改变前为{best_avg_height}，改变后为{avg_height}")



    if enable_limited_flag and y_dump_truck_flag and not x_dump_truck_flag:
        avg_height += limited_change_height
        logger.info(f"海陆侧甩斗，高度增加{limited_change_height}，改变前为{avg_height-limited_change_height}，改变后为{avg_height}")
        if avg_height>min(above_current_line_layer2_min_height,limited_height):
            avg_height=min(above_current_line_layer2_min_height,limited_height)
            logger.info(f"超过限制高度，高度设置为{min(above_current_line_layer2_min_height,limited_height)}")

    else:
        if avg_height >= above_current_line_layer_min_height:
            avg_height = above_current_line_layer_min_height
            logger.info(f"高于大车方向甩斗限制的最低高度，高度设置为{above_current_line_layer_min_height}")

    # 计算抓取点处于哪一层
    capture_point_layer = math.ceil(abs(hatch_height - avg_height) / floor_height)
    
    # 创建抓取点字典
    capture_point = {'x': float(avg_x), 'y': float(avg_y), 'z': float(avg_height)}
    if enable_limited_flag and y_dump_truck_flag and not x_dump_truck_flag:
        logger.info(f"未处理前抓取点的坐标: X={avg_x:.3f}, Y={avg_y:.3f}, Z={max_height:.3f}")
    else:
        logger.info(f"未处理前抓取点的坐标: X={avg_x:.3f}, Y={avg_y:.3f}, Z={best_avg_height:.3f}")

    logger.info(f"抓取点的坐标: X={avg_x:.3f}, Y={avg_y:.3f}, Z={avg_height:.3f}")
    
    return capture_point,capture_point_layer


def get_bottom_depth_result(process_id="160", current_hatch=None):
    """
    读取measure_deepth.py的测量结果
    
    参数:
        process_id: 进程标识符，默认为"160"
        current_hatch: 当前舱口号，如果为None则尝试读取最新的结果文件
    
    返回:
        dict: 包含舱底深度等信息的字典，如果读取失败返回None
    """
    # 确定要读取的文件路径
    if current_hatch is not None:
        # 指定了舱口号，直接构造文件路径
        result_file_path = os.path.join(os.path.dirname(__file__), f"depth_measurement_result_{process_id}_{current_hatch}.json")
    else:
        # 没有指定舱口号，查找最新的结果文件
        try:
            import glob
            base_dir = os.path.dirname(__file__)
            pattern = f"depth_measurement_result_{process_id}_*.json"
            matching_files = glob.glob(os.path.join(base_dir, pattern))
            
            if not matching_files:
                logger.warning(f"未找到进程 {process_id} 的舱底深度测量结果文件")
                return None
            
            # 找到最新的文件
            result_file_path = max(matching_files, key=os.path.getmtime)
        except Exception as e:
            logger.exception(f"查找舱底深度测量结果文件失败: {e}", exc_info=True)
            return None
    
    # 读取文件
    try:
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 检查结果是否是最近的（5分钟内）
            current_time = time.time()
            if current_time - result_data.get('timestamp', 0) < 300:  # 5分钟
                logger.info(f"读取到舱底深度测量结果: {result_data['bottom_depth']:.2f}米")
                return result_data
            else:
                logger.warning("舱底深度测量结果过期")
                return None
        else:
            logger.warning(f"舱底深度测量结果文件不存在: {result_file_path}")
            return None
    except Exception as e:
        logger.exception(f"读取舱底深度测量结果失败: {e}", exc_info=True)
        return None

#定义一个函数，用来获取下一条线
def get_next_line(current_line, direction, line_numbers, tried_lines=None):
  line_numbers = sorted(line_numbers)  # 确保从小到大
  min_line = line_numbers[0]
  max_line = line_numbers[-1]
  total_lines = len(line_numbers)
  
  if tried_lines is None:
      tried_lines = set()
  
  # 如果总线数大于3条，使用新的换线策略
  if total_lines > 3:
      # 最左边的线的右边第一条线（第2条线）优先向左侧换线
      if current_line == min_line + 1 and (current_line - 1) not in tried_lines:  # 第2条线且左边未尝试
          next_line = current_line - 1
          direction = -1
      # 最右边的线的左边第一条线（倒数第2条线）优先向右侧换线
      elif current_line == max_line - 1 and (current_line + 1) not in tried_lines:  # 倒数第2条线且右边未尝试
          next_line = current_line + 1
          direction = 1
      else:
          # 其他情况或优先方向已尝试过，使用原有逻辑
          next_line = current_line + direction
          # 到达边界时反向
          if next_line < min_line or next_line > max_line:
              direction *= -1
              next_line = current_line + direction
  else:
      # 总线数不大于3条时，保持原有逻辑
      next_line = current_line + direction
      # 到达边界时反向
      if next_line < min_line or next_line > max_line:
          direction *= -1
          next_line = current_line + direction

  return next_line, direction