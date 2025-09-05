# 导入所需的库
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
log_file_path = os.path.join(log_dir, "coal_pile_segmentation.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    filename=log_file_path,   # 指定日志文件路径
    filemode="a",              # "a" 追加写入, "w" 覆盖写入
    encoding="utf-8"           # 保证中文正常输出
)
logger = logging.getLogger(__name__)


def cluster_points_dbscan(points, eps=0.6, min_samples=5, visualize_hdbscan=False):
    """
    
    Args:
        points (np.array): 点云数组 (N, 3)
        eps_equivalent (float): 等效的eps值
        min_samples (int): 最小样本数
        visualize_hdbscan (bool): 是否可视化
    Returns:
        np.array: 聚类标签
    """
    if len(points) == 0:
        logging.error("输入点云为空，跳过HDBSCAN聚类。")
        return np.array([])
        
    # print(f"处理 {len(points):,} 个点")
    
    # 关键参数设置
    clusterer = hdbscan.HDBSCAN(
        # 核心参数：模拟DBSCAN的eps效果
        cluster_selection_epsilon=eps,  # 等效eps
        
        # 最小样本数：与DBSCAN相同
        min_samples=min_samples,
        
        # 最小聚类大小：通常设为min_samples的2-3倍
        min_cluster_size=min_samples * 2,  
        
        # 算法选择
        algorithm='best',  # 自动选择最优算法
        
        # 聚类选择方法
        cluster_selection_method='eom',  # 超额质量方法，更稳定
        
        # 性能优化
        core_dist_n_jobs=-1,  # 并行计算核心距离
        
        # 其他重要参数
        allow_single_cluster=False,  # 不允许单一聚类
        prediction_data=True  # 启用预测数据，用于后续分析
    )
    
    start_time = time.time()
    labels = clusterer.fit_predict(points.astype(np.float32))
    end_time = time.time()
    
    # 统计结果
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    #统计有多少噪音点
    n_noise = np.sum(labels == -1)
    
    # print(f"HDBSCAN完成，耗时: {end_time - start_time:.4f}秒")
    # print(f"发现 {n_clusters} 个聚类，{n_noise:,} 个噪声点")
    # print(f"噪声比例: {n_noise/len(points)*100:.1f}%")
    
    if visualize_hdbscan:
        visualize_clustered_pcd(points, labels, 
                              title=f"HDBSCAN聚类 (等效eps={eps}, {n_clusters}个聚类)")
    return labels
# --- 分割煤堆 ---
def segment_coal_pile(original_points, hatch_corners_refined):

  
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
        return np.array([])
    
    if original_points.shape[1] < 4:
        return np.array([])
    
    # 1. 计算船舱四个顶点的x坐标平均值
    avg_x_corners = np.mean(hatch_corners_refined[:, 0])
    
    # 定义X坐标分割阈值：x坐标平均值加7米
    x_split_threshold = avg_x_corners + 7
    
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
            
            shrink_amount = 0.6
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
        return np.array([])
    
    # HDBSCAN聚类参数
    hdbscan_eps_coal = 1
    hdbscan_min_samples_coal = 20
    
    # 执行聚类（只使用XYZ坐标）
    labels_coal_cluster = cluster_points_dbscan(combined_points_for_clustering[:, :3], 
                                                eps=hdbscan_eps_coal, 
                                                min_samples=hdbscan_min_samples_coal,visualize_hdbscan=False)
    
    # 找到点数最多的聚类（排除噪声点-1）
    unique_labels_coal, counts_coal = np.unique(labels_coal_cluster[labels_coal_cluster != -1], return_counts=True)
    
    if len(unique_labels_coal) == 0:
        return np.array([])
    
    # 返回最大聚类的点
    largest_cluster_label_coal = unique_labels_coal[np.argmax(counts_coal)]
    final_coal_cluster_points = combined_points_for_clustering[labels_coal_cluster == largest_cluster_label_coal]
    
    return final_coal_cluster_points

# WebSocket服务器配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 27052
SERVER_PATH = "/point-cloud"
USER_ID = "coal-pile-detector"
URI = f"ws://{SERVER_HOST}:{SERVER_PORT}{SERVER_PATH}?userId={USER_ID}"

# 定义二进制数据解析格式
# 头部格式：魔数(4) + 版本(2) + 头长度(2) + 点大小(2) + 时间戳类型(2) + 帧ID(4) + 作业类型(4) + 大车当前位置(8) + 当前作业舱口(4) + 当前执行步骤(4) + 开始时间戳(8) + 结束时间戳(8) + 包数量(4) + 点数量(4) + 舱口坐标系四个角点坐标(96) + 世界坐标系四个角点坐标(96)
HEADER_FORMAT = "<4s HHHHIIQ II QQ II 12d 12d"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
POINT_FORMAT = "<iiiBB"
POINT_SIZE = struct.calcsize(POINT_FORMAT)


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
        while attempts < self.max_reconnect_attempts:
            try:
                # print(f"尝试连接到 WebSocket 服务器: {self.uri} (第{attempts + 1}次)")
                self.websocket = await websockets.connect(self.uri, max_size=8 * 1024 * 1024)
                self.is_connected = True
                print("WebSocket连接成功！")
                return True
            except Exception as e:
                attempts += 1
                # print(f"WebSocket连接失败: {e}")
                if attempts < self.max_reconnect_attempts:
                    # print(f"等待 {self.reconnect_delay} 秒后重试...")
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

# 全局WebSocket管理器实例
ws_manager = None

async def parse_point_cloud_data(data: bytes):
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
                current_machine_position,
                current_hatch, current_step, start_ts_raw, end_ts_raw, pkt_count, num_points,
                # 四个角点坐标 (每个角点3个double值：x, y, z)
                corner1_x, corner1_y, corner1_z,
                corner2_x, corner2_y, corner2_z,
                corner3_x, corner3_y, corner3_z,
                corner4_x, corner4_y, corner4_z,
                # 世界坐标系四个角点坐标 (每个角点3个double值：x, y, z)
                world_corner1_x, world_corner1_y, world_corner1_z,
                world_corner2_x, world_corner2_y, world_corner2_z,
                world_corner3_x, world_corner3_y, world_corner3_z,
                world_corner4_x, world_corner4_y, world_corner4_z
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
    print(f"  Current Machine Position: {current_machine_position}")
    print(f"  Corner1: ({corner1_x:.3f}, {corner1_y:.3f}, {corner1_z:.3f})")
    print(f"  Corner2: ({corner2_x:.3f}, {corner2_y:.3f}, {corner2_z:.3f})")
    print(f"  Corner3: ({corner3_x:.3f}, {corner3_y:.3f}, {corner3_z:.3f})")
    print(f"  Corner4: ({corner4_x:.3f}, {corner4_y:.3f}, {corner4_z:.3f})")
    print(f"  World Corner1: ({world_corner1_x:.3f}, {world_corner1_y:.3f}, {world_corner1_z:.3f})")
    print(f"  World Corner2: ({world_corner2_x:.3f}, {world_corner2_y:.3f}, {world_corner2_z:.3f})")
    print(f"  World Corner3: ({world_corner3_x:.3f}, {world_corner3_y:.3f}, {world_corner3_z:.3f})")
    print(f"  World Corner4: ({world_corner4_x:.3f}, {world_corner4_y:.3f}, {world_corner4_z:.3f})")
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
            'current_machine_position': current_machine_position,
            'current_hatch': current_hatch,
            'current_step': current_step,
            'num_points': num_points,
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
        },
      
        'points': points
    }



async def get_point_cloud_from_websocket_persistent():
    """
    从持久WebSocket连接获取点云数据并转换为numpy数组格式。
    """
    global ws_manager
    
    if not ws_manager:
        print("WebSocket管理器未初始化")
        return None
    
    # print("等待接收点云数据...")
    
    while True:
        message = await ws_manager.receive_message()
        if message is None:
            print("接收消息失败")
            return None
        
        # 检查是否为二进制数据
        if isinstance(message, bytes):
            # print(f"接收到二进制点云数据，长度: {len(message)} 字节")
            
            try:
                # 直接解析二进制点云数据
                parsed_data = await parse_point_cloud_data(message)
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
                        parsed_data = await parse_point_cloud_data(decoded_bytes)
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


async def send_coal_pile_to_websocket_persistent(coal_pile_points, header_info, hatch_corners_refined):
    """
    通过持久WebSocket连接发送煤堆点云数据到服务器
    
    Args:
        coal_pile_points (np.array): 煤堆点云数据，形状为 (N, 4) 包含X,Y,Z,Intensity
        header_info (dict): 原始头部信息
        hatch_corners_refined (np.array): 船舱口四个顶点坐标，形状为 (4, 3)
    
    Returns:
        bool: 发送是否成功
    """
    global ws_manager
    
    if not ws_manager:
        print("WebSocket管理器未初始化")
        return False
    
    try:
        print("准备发送煤堆数据...")
        
        # 构造要发送的数据
        # data_to_send = {
        #     "type": "coal_pile",
        #     "timestamp": int(time.time() * 1000),  # 毫秒时间戳
        #     "frame_id": header_info.get('frame_id', 0),
        #     "current_hatch": header_info.get('current_hatch', 0),
        #     "current_step": 6,  # 煤堆分割完成步骤
        #     "detection_success": len(coal_pile_points) > 0,
        #     "point_count": len(coal_pile_points),
        #     "hatch_corners": {
        #         "corner1": {"x": float(hatch_corners_refined[0][0]), "y": float(hatch_corners_refined[0][1]), "z": float(hatch_corners_refined[0][2])},
        #         "corner2": {"x": float(hatch_corners_refined[1][0]), "y": float(hatch_corners_refined[1][1]), "z": float(hatch_corners_refined[1][2])},
        #         "corner3": {"x": float(hatch_corners_refined[2][0]), "y": float(hatch_corners_refined[2][1]), "z": float(hatch_corners_refined[2][2])},
        #         "corner4": {"x": float(hatch_corners_refined[3][0]), "y": float(hatch_corners_refined[3][1]), "z": float(hatch_corners_refined[3][2])}
        #     }
        # }
        data_to_send = {
            "type": "coal_pile",
            "timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "frame_id": header_info.get('frame_id', 0),
            "current_hatch": header_info.get('current_hatch', 0),
            "point_count": len(coal_pile_points),
            "hatch_corners": {
                "corner1": {"x": float(hatch_corners_refined[0][0]), "y": float(hatch_corners_refined[0][1]), "z": float(hatch_corners_refined[0][2])},
                "corner2": {"x": float(hatch_corners_refined[1][0]), "y": float(hatch_corners_refined[1][1]), "z": float(hatch_corners_refined[1][2])},
                "corner3": {"x": float(hatch_corners_refined[2][0]), "y": float(hatch_corners_refined[2][1]), "z": float(hatch_corners_refined[2][2])},
                "corner4": {"x": float(hatch_corners_refined[3][0]), "y": float(hatch_corners_refined[3][1]), "z": float(hatch_corners_refined[3][2])}
            }
        }
        
        # 如果检测到煤堆，添加点云数据
        if len(coal_pile_points) > 0:
            # 转换点云数据为列表格式
            points_list = []
            for point in coal_pile_points:
                points_list.append({
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    # "intensity": float(point[3]) if len(point) > 3 else 0.0
                })
            
            data_to_send["coal_pile_points"] = points_list
        
        # 转换为JSON字符串并发送
        json_message = json.dumps(data_to_send, ensure_ascii=False)
        success = await ws_manager.send_message(json_message)
        
        if success:
            if len(coal_pile_points) > 0:
                print(f"煤堆数据已发送到WebSocket服务器，包含 {len(coal_pile_points)} 个点")
            else:
                print("空煤堆结果已发送到WebSocket服务器")
            # print(f"发送的数据类型: {data_to_send['type']}")
            return True
        else:
            print("发送煤堆数据失败")
            return False
            
    except Exception as e:
        print(f"发送煤堆数据到WebSocket失败: {e}")
        logger.error(f"WebSocket发送煤堆数据失败: {e}")
        return False

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
            max_size=10 * 1024 * 1024,  # 10MB最大消息大小
            ping_interval=20,  # 20秒ping间隔
            ping_timeout=10    # 10秒ping超时
        )
        
        logger.info(f"广播服务器已启动，等待客户端连接...")
        return self.server

# 全局广播服务器实例
broadcast_server = None

async def main():
    global ws_manager, broadcast_server
    
    # 初始化WebSocket管理器
    ws_manager = WebSocketManager(URI,5,1)
    # 初始化广播服务器
    broadcast_server = CoalPileBroadcastServer(host="127.0.0.1", port=8765)
    
 # 启动广播服务器
    try:
        await broadcast_server.start_server()
        print("广播服务器已启动")
    except Exception as e:
        logger.error(f"启动广播服务器失败: {e}")
        return
    
    
    # 建立初始连接
    if not await ws_manager.connect():
        print("无法建立WebSocket连接，程序退出")
        return
    
    try:
        USE_CLUSTERING = False    #控制是否使用聚类
        VIS_ORIGINAL_3D = False  # 可视化原始点云
        VISUALIZE_COARSE_FILTRATION = False   # 可视化粗过滤结果
        VISUALIZE_HDBSCAN = False      # 可视化HDBSCAN聚类结果
        VISUALIZE_RECTANGLE = False  # 可视化矩形检测过程
        VISUALIZE_FINAL_RESULT = False  # 可视化最终结果

        while True:
            # print("=== 从WebSocket获取点云数据 ===")
            original_points, header_info = await get_point_cloud_from_websocket_persistent()
            
            if original_points is None or len(original_points) == 0:
                # print("错误: 无法从WebSocket获取点云数据，等待下一次尝试...")
                await asyncio.sleep(0.5)  # 等待0.5秒后重试
                continue
            current_hatch = header_info.get('current_hatch', 0)
            current_step = header_info.get('current_step', 0)
            hatch_corners = header_info.get('hatch_corners', {})
            if current_step != 5:
                # print("当前步骤不是5，跳过处理")    
                continue
             # 将角点数据转换为numpy数组格式 (4, 3)
            hatch_corners_refined = np.array([
                [hatch_corners['corner1']['x'], hatch_corners['corner1']['y'], hatch_corners['corner1']['z']],
                [hatch_corners['corner2']['x'], hatch_corners['corner2']['y'], hatch_corners['corner2']['z']],
                [hatch_corners['corner3']['x'], hatch_corners['corner3']['y'], hatch_corners['corner3']['z']],
                [hatch_corners['corner4']['x'], hatch_corners['corner4']['y'], hatch_corners['corner4']['z']]
            ], dtype=np.float32)

            coal_pile_points = segment_coal_pile(original_points, hatch_corners_refined)
            if len(coal_pile_points) > 0:
                # print(f"成功分割出煤堆，包含 {len(coal_pile_points)} 个点")
                 # 构造煤堆数据
                coal_pile_data = {
                    "frame_id": header_info.get('frame_id', 0),
                    "current_hatch": header_info.get('current_hatch', 0),
                    "point_count": len(coal_pile_points),
                    "detection_success": len(coal_pile_points) > 0,
                    "hatch_corners": {
                        "corner1": {"x": float(hatch_corners_refined[0][0]), "y": float(hatch_corners_refined[0][1]), "z": float(hatch_corners_refined[0][2])},
                        "corner2": {"x": float(hatch_corners_refined[1][0]), "y": float(hatch_corners_refined[1][1]), "z": float(hatch_corners_refined[1][2])},
                        "corner3": {"x": float(hatch_corners_refined[2][0]), "y": float(hatch_corners_refined[2][1]), "z": float(hatch_corners_refined[2][2])},
                        "corner4": {"x": float(hatch_corners_refined[3][0]), "y": float(hatch_corners_refined[3][1]), "z": float(hatch_corners_refined[3][2])}
                    }
                }
                # 添加煤堆点云数据
                points_list = []
                for point in coal_pile_points:
                    points_list.append({
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                        "intensity": float(point[3]) if len(point) > 3 else 0.0
                    })
                coal_pile_data["coal_pile_points"] = points_list
                
                # 广播煤堆数据到所有客户端
                await broadcast_server.broadcast_coal_pile_data(coal_pile_data)
                print(f"煤堆数据已广播到所有连接的客户端")
                # 发送煤堆数据到WebSocket服务器
            else:
                # print("未检测到煤堆")
                logger.info("未检测到煤堆")
            
    except KeyboardInterrupt:
        # print("\n程序被用户中断")
        logger.info("程序被用户中断")
    except Exception as e:
        # print(f"程序运行出错: {e}")
        logger.error(f"程序运行出错: {e}")
    finally:
        # 确保在程序结束时关闭WebSocket连接
        if ws_manager:
            await ws_manager.disconnect()




if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # print("\n程序已手动停止。")
        logger.info("程序已手动停止。")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")


