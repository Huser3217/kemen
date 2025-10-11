


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
import config
from tools import visualize_pcd_3d


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)


# 禁用websockets库的日志
logging.getLogger('websockets').setLevel(logging.CRITICAL)
logging.getLogger('websockets.client').setLevel(logging.CRITICAL)
logging.getLogger('websockets.server').setLevel(logging.CRITICAL)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "all_hatch.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    filename=log_file_path,   # 指定日志文件路径
    filemode="a",              # "a" 追加写入, "w" 覆盖写入
    encoding="utf-8"           # 保证中文正常输出
)
logger = logging.getLogger(__name__)



# 根据Java服务器代码调整的格式定义
HEADER_FORMAT = "<4s HH III"  # 魔数(4s) + 版本(H) + 头部长度(H) + 帧ID(I) + 点数量(I) + 时间戳(I)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
POINT_FORMAT = "<ffff" 
POINT_SIZE = struct.calcsize(POINT_FORMAT)

# WebSocket服务器配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 27052
SERVER_PATH = "/leishen-point-cloud"
USER_ID = "all-hatch-detector"
URI = f"ws://{SERVER_HOST}:{SERVER_PORT}{SERVER_PATH}?userId={USER_ID}"



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
                magic, version, header_len, frame_id, num_points, time_stamp
            ) = header_data
    except struct.error as e:
        print(f"解析头部失败: {e}")
        return None

    # 验证魔术字
    if magic != b'LSPC':
        logger.warning(f"魔术字不匹配！预期 b'LSPC'，实际 {magic}")
        return None

    print(f"\n--- 解析头部 v{version} ---")
    print(f"  Magic: {magic.decode()}")
    print(f"  Version: {version}")
    print(f"  Header Length: {header_len}")
    print(f"  Frame ID: {frame_id}")
    print(f"  Number of Points: {num_points}")
    print(f"  Time Stamp: {time_stamp}")

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
            x, y, z, intensity = struct.unpack(POINT_FORMAT, point_bytes)
            
            points.append({'x': x, 'y': y, 'z': z, 'refl': intensity, 'tag': 0})
        except struct.error as e:
            logger.debug(f"解析点数据失败，点索引 {i}: {e}")
            break

    # print(f"成功解析 {len(points)} 个点。")
    return {
        'header': {
            'version': version,
            'frame_id': frame_id,
            'num_points': num_points,
            'time_stamp': time_stamp,
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





async def process_point_cloud(original_points,is_visualize=False):
    """
    处理从WebSocket获取的点云数据
    
    Args:
        original_points: numpy数组，形状为(N, 4)，包含X, Y, Z, Intensity
    
    Returns:
        dict: 包含处理结果的字典
    """
    try:
        # 提取XYZ坐标（忽略反射强度）
        points = original_points[:, :3]
        
        if len(points) < 10:
            print("点云数据太少，无法进行处理")
            return None
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print(f"开始处理点云数据，共 {len(points)} 个点")
        
        # 可视化原始点云
        if is_visualize:
            print("显示原始点云...")
            o3d.visualization.draw_geometries([pcd], window_name="原始点云")
        
        # 第一次平面拟合
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=50000)
        print(f"第一次平面方程: {plane_model[0]:.2f}x + {plane_model[1]:.2f}y + {plane_model[2]:.2f}z + {plane_model[3]:.2f} = 0")
        
        # 获取平面上的点
        plane_points = points[inliers]
        
        # 可视化第一次拟合的平面点云
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
        plane_pcd.paint_uniform_color([1, 0, 0])  # 红色
        if is_visualize:
            print("显示第一次拟合的平面点云...")
            o3d.visualization.draw_geometries([plane_pcd], window_name="第一次拟合平面")
        
        # 计算第一次平面上点的y坐标平均值
        y_avg = np.mean(plane_points[:, 1])
        print(f"平面1上点的y坐标平均值: {y_avg:.2f}")
        
        # 去除平面上的点云
        remaining_points = points[np.setdiff1d(np.arange(len(points)), inliers)]
        
        if len(remaining_points) < 10:
            print("剩余点云数据太少，无法继续处理")
            return {"plane1_y_avg": y_avg, "clusters": []}
        
        # 可视化剩余点云
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)
        remaining_pcd.paint_uniform_color([0, 1, 0])  # 绿色
        if is_visualize:
            print("显示剩余点云...")
            o3d.visualization.draw_geometries([remaining_pcd], window_name="剩余点云")
        
        # 第二次平面拟合
        if len(remaining_points) > 3:
            plane_model2, inliers2 = remaining_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=50000)
            print(f"第二次平面方程: {plane_model2[0]:.2f}x + {plane_model2[1]:.2f}y + {plane_model2[2]:.2f}z + {plane_model2[3]:.2f} = 0")
            
            # 可视化第二次拟合的平面点云
            plane_points2 = remaining_points[inliers2]
            plane_pcd2 = o3d.geometry.PointCloud()
            plane_pcd2.points = o3d.utility.Vector3dVector(plane_points2)
            plane_pcd2.paint_uniform_color([0, 0, 1])  # 蓝色
            if is_visualize:
                print("显示第二次拟合的平面点云...")
                o3d.visualization.draw_geometries([plane_pcd2], window_name="第二次拟合平面")
            
            # 计算第二次拟合的平面上点的y坐标平均值
            y_avg2 = np.mean(plane_points2[:, 1])
            print(f"平面2上点的y坐标平均值: {y_avg2:.2f}")
            
            # 比较两次拟合的平面上点的y坐标平均值，选择y值更小的平面进行聚类
            if y_avg2 < y_avg:
                cluster_pcd = plane_pcd2
                selected_plane = "平面2"
            else:
                cluster_pcd = plane_pcd
                selected_plane = "平面1"
            
            print(f"选择{selected_plane}进行聚类分析")
            
            # DBSCAN聚类
            dbscan_labels = np.array(cluster_pcd.cluster_dbscan(eps=0.4, min_points=10))
            max_label = dbscan_labels.max()
            print(f"聚类数: {max_label + 1}")
            
            # 可视化聚类结果
            colors = plt.get_cmap("tab20")(dbscan_labels / (max_label if max_label > 0 else 1))
            colors[dbscan_labels < 0] = 0
            cluster_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            if is_visualize:
                print("显示聚类结果...")
                o3d.visualization.draw_geometries([cluster_pcd], window_name="聚类结果")
            
            # 选取点数最多的四个聚类结果
            unique_labels, counts = np.unique(dbscan_labels, return_counts=True)
            valid_mask = unique_labels >= 0
            valid_labels = unique_labels[valid_mask]
            valid_counts = counts[valid_mask]
            
            if len(valid_labels) == 0:
                print("没有找到有效的聚类")
                return {"plane1_y_avg": y_avg, "plane2_y_avg": y_avg2, "clusters": []}
            
            # 按点数从大到小排序，选出前四个标签
            top_indices = np.argsort(-valid_counts)[:4]
            top_labels = valid_labels[top_indices]
            
            # 获取对应的聚类点云
            valid_clusters = [cluster_pcd.select_by_index(np.where(dbscan_labels == label)[0]) for label in top_labels]
            
            # 计算每个聚类的x坐标范围并筛选
            filtered_clusters = []
            cluster_x_stats = []
            
            for i, cluster in enumerate(valid_clusters):
                cluster_points = np.asarray(cluster.points)
                if len(cluster_points) >= 100:
                    # 计算x坐标范围
                    x_sorted = np.sort(cluster_points[:, 0])
                    x_avg_min = np.mean(x_sorted[:50])
                    x_avg_max = np.mean(x_sorted[-50:])
                    x_range = x_avg_max - x_avg_min
                    x_center = np.mean(cluster_points[:, 0])
                    
                    print(f"聚类 {i}: x_center = {x_center:.2f}, x_avg_min = {x_avg_min:.2f}, x_avg_max = {x_avg_max:.2f}, x_range = {x_range:.2f}")
                    
                    # 筛选出x坐标范围大于16米的聚类
                    if x_range > 16:
                        filtered_clusters.append(cluster)
                        cluster_x_stats.append([x_center, x_avg_min, x_avg_max, x_range])
            
            if filtered_clusters:
                print(f"筛选后保留 {len(filtered_clusters)} 个聚类")
                if is_visualize:
                    print("显示筛选后的聚类...")
                    o3d.visualization.draw_geometries(filtered_clusters, window_name="筛选后的聚类")
                
                # 输出最终统计信息
                print("\n=== 最终聚类统计信息 ===")
                for i, stats in enumerate(cluster_x_stats):
                    print(f"聚类 {i}: x_center = {stats[0]:.2f}, x_avg_min = {stats[1]:.2f}, x_avg_max = {stats[2]:.2f}, x_range = {stats[3]:.2f}")
            else:
                print("没有聚类满足x坐标范围大于16米的条件")
            
            return {
                "cluster_stats": cluster_x_stats
            }
        else:
            logger.warning("剩余点云数据不足，无法进行第二次平面拟合",exc_info=True)    
            return None
            
    except Exception as e:
        logger.error(f"点云处理过程中出错: {e}", exc_info=True)
        return None

async def main():
    # 初始化WebSocket管理器
    ws_manager = WebSocketManager(URI, -1, 1)
    # 建立初始连接
    await ws_manager.connect()
    is_visualize = False
    try:
        while True:
            try:
                if not ws_manager.is_connected:
                    print("WebSocket断开，重连中...")
                    await ws_manager.connect()
                original_points, header_info = await get_point_cloud_from_websocket_persistent(ws_manager, HEADER_SIZE, POINT_SIZE, HEADER_FORMAT, POINT_FORMAT)
                
                if original_points is None or len(original_points) == 0:
                    await asyncio.sleep(0.5)  # 等待0.5秒后重试
                    continue
                
                

                print("\n开始进行点云处理分析...")
                result = await process_point_cloud(original_points, is_visualize)
                
                if result:
                    print(f"\n=== 处理完成 ===")
                    print(f"处理结果: {result}")
                else:
                    print("点云处理失败")
                    
            except Exception as e:
                logger.warning("出现问题", exc_info=True)

    except Exception as e:
        logger.error(f"主循环出错: {e}", exc_info=True)
    finally:
        await ws_manager.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # print("\n程序已手动停止。")
        logger.info("程序已手动停止。")
    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info= True)
