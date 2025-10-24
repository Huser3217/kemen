
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

from tools import *
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
log_file_path = os.path.join(log_dir, "hatch_coal_170_zs.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    filename=log_file_path,   # 指定日志文件路径
    filemode="a",              # "a" 追加写入, "w" 覆盖写入
    encoding="utf-8"           # 保证中文正常输出
)
logger = logging.getLogger(__name__)







# 定义二进制数据解析格式
# 头部格式：魔数(4) + 版本(2) + 头长度(2) + 点大小(2) + 时间戳类型(2) + 帧ID(4) + 作业类型(4) + 上次大车位置(8) + 大车当前位置(8) + 当前作业舱口(4) + 计算信号(4) + 开始时间戳(8) + 结束时间戳(8) + 包数量(4) + 点数量(4) + 舱口坐标系四个角点坐标(96) + 世界坐标系四个角点坐标(96)
HEADER_FORMAT = config.HEADER_FORMAT
HEADER_SIZE = config.HEADER_SIZE
POINT_FORMAT = config.POINT_FORMAT
POINT_SIZE = config.POINT_SIZE

# WebSocket服务器配置
SERVER_HOST = config.GrabPointCalculationConfig_170.SERVER_HOST
SERVER_PORT = config.GrabPointCalculationConfig_170.SERVER_PORT
SERVER_PATH = config.GrabPointCalculationConfig_170.SERVER_PATH
USER_ID = config.GrabPointCalculationConfig_170.USER_ID
URI = f"ws://{SERVER_HOST}:{SERVER_PORT}{SERVER_PATH}?userId={USER_ID}"


async def main():

    # 初始化WebSocket管理器
    ws_manager = WebSocketManager(URI,-1,1)
    # 初始化广播服务器
    broadcast_server = CoalPileBroadcastServer(host=config.GrabPointCalculationConfig_170.BROADCAST_HOST, port=config.GrabPointCalculationConfig_170.BROADCAST_PORT)
    
 # 启动广播服务器
    try:
        await broadcast_server.start_server()
        print("广播服务器已启动")
    except Exception as e:
        logger.error(f"启动广播服务器失败: {e}",exc_info= True)
        return
    
    
    # 建立初始连接
    await ws_manager.connect()
    
    # 定义一个变量来存储换线方向
    # 1: 沿X轴正方向换线 (编号增加的方向)
    # -1: 沿X轴负方向换线 (编号减少的方向)
    persisted_line_change_direction = -1 



    try:



        while True:
            try:

                IS_FIRST_TIME = False
                # 重置换线变量
                tried_lines = set()
                total_lines = 0

                if not ws_manager.is_connected:
                    print("WebSocket断开，重连中...")
                    await ws_manager.connect()

                # print("=== 从WebSocket获取点云数据 ===")             
                original_points, header_info = await get_point_cloud_from_websocket_persistent(ws_manager,HEADER_SIZE,POINT_SIZE,HEADER_FORMAT,POINT_FORMAT)
                
                if original_points is None or len(original_points) == 0:
                    await asyncio.sleep(0.5)  # 等待0.5秒后重试
                    continue

                importlib.reload(config)


                VIS_ORIGINAL_3D = config.GrabPointCalculationConfig_170.VIS_ORIGINAL_3D  # 可视化原始点云
                VISUALIZE_COARSE_FILTRATION = config.GrabPointCalculationConfig_170.VISUALIZE_COARSE_FILTRATION   # 可视化粗过滤结果
                VISUALIZE_RECTANGLE = config.GrabPointCalculationConfig_170.VISUALIZE_RECTANGLE  # 可视化矩形检测过程
                VISUALIZE_FINAL_RESULT = config.GrabPointCalculationConfig_170.VISUALIZE_FINAL_RESULT      # 可视化最终结果
                visualize_clustered_pcd = config.GrabPointCalculationConfig_170.visualize_clustered_pcd  # 可视化聚类后的结果
                visualize_coal_points = config.GrabPointCalculationConfig_170.visualize_coal_points  # 可视化煤堆点
                # 安装参数
                translation = [1.58, 24.324, 31.4]
                    
                # 旋转角度 [roll, pitch, yaw] 
                rotation_angles = [-6.05, 0.15, 0]
                
                current_hatch = header_info.get('current_hatch', 0)
                current_step = header_info.get('current_step', 0)
                hatch_corners = header_info.get('hatch_corners', {})
                world_corners = header_info.get('world_corners', {})
                is_detection_hatch = header_info.get('is_detection_hatch', 0)
                last_machine_position = header_info.get('last_machine_position', 0) 
                current_machine_position = header_info.get('current_machine_position', {})
                last_capture_point = header_info.get('last_capture_point', {})
                mode_flag = header_info.get('mode_flag', 0)
                
                last_capture_point_x = last_capture_point.get('x', 0)
                last_capture_point_y = last_capture_point.get('y', 0)
                last_capture_point_z = last_capture_point.get('z', 0)
                
                
                if current_step != 1:
                    # print("当前计算信号不是1，跳过处理")    
                    continue
                 # 将角点数据转换为numpy数组格式 (4, 3)
                hatch_corners_refined = np.array([
                    [hatch_corners['corner1']['x'], hatch_corners['corner1']['y'], hatch_corners['corner1']['z']],
                    [hatch_corners['corner2']['x'], hatch_corners['corner2']['y'], hatch_corners['corner2']['z']],
                    [hatch_corners['corner3']['x'], hatch_corners['corner3']['y'], hatch_corners['corner3']['z']],
                    [hatch_corners['corner4']['x'], hatch_corners['corner4']['y'], hatch_corners['corner4']['z']]
                ], dtype=np.float32)
                
                world_coords_refined = np.array([
                    [world_corners['corner1']['x'], world_corners['corner1']['y'], world_corners['corner1']['z']],
                    [world_corners['corner2']['x'], world_corners['corner2']['y'], world_corners['corner2']['z']],
                    [world_corners['corner3']['x'], world_corners['corner3']['y'], world_corners['corner3']['z']],
                    [world_corners['corner4']['x'], world_corners['corner4']['y'], world_corners['corner4']['z']]
                ], dtype=np.float32)
                
                  #如果这四个顶点坐标都为（0，0，0），说明这是第一次识别
                if hatch_corners_refined[0][0] == 0 and hatch_corners_refined[0][1] == 0 and hatch_corners_refined[0][2] == 0 and \
                   hatch_corners_refined[1][0] == 0 and hatch_corners_refined[1][1] == 0 and hatch_corners_refined[1][2] == 0 and \
                   hatch_corners_refined[2][0] == 0 and hatch_corners_refined[2][1] == 0 and hatch_corners_refined[2][2] == 0 and \
                   hatch_corners_refined[3][0] == 0 and hatch_corners_refined[3][1] == 0 and hatch_corners_refined[3][2] == 0:
                
                
                    IS_FIRST_TIME=True
                    radar_center_x,radar_center_y,radar_center_z=0, 0,0
                
                else:
                    #矩形的中心世界坐标转为雷达坐标系中
                   
                    hatch_center_x=(hatch_corners_refined[0][0]+hatch_corners_refined[1][0]+hatch_corners_refined[2][0]+hatch_corners_refined[3][0])/4
                    hatch_center_y=(hatch_corners_refined[0][1]+hatch_corners_refined[1][1]+hatch_corners_refined[2][1]+hatch_corners_refined[3][1])/4
                    hatch_center_z=(hatch_corners_refined[0][2]+hatch_corners_refined[1][2]+hatch_corners_refined[2][2]+hatch_corners_refined[3][2])/4
                    hatch_point = np.array([
                        [hatch_center_x,hatch_center_y,hatch_center_z]
                    ])
                    
                    world_center_result = lidar_to_world_with_x(hatch_point,translation,0,rotation_angles)
                    world_center_x, world_center_y, world_center_z = world_center_result[0][0], world_center_result[0][1], world_center_result[0][2]
                    # world_center_x,world_center_y,world_center_z = lidar_to_world_with_x(hatch_point,translation,0,rotation_angles)[0]
                    world_center_x = world_center_x+(last_machine_position-current_machine_position)
                    world_point = np.array([
                        [world_center_x,world_center_y,world_center_z]
                    ])
                    #------------------------------------------------------------------------------------------------------
                    
                    radar_center_result = world_to_lidar_with_x(world_point,translation,rotation_angles)
                    radar_center_x, radar_center_y, radar_center_z = radar_center_result[0][0], radar_center_result[0][1], radar_center_result[0][2]
                     
                
                if VIS_ORIGINAL_3D:
                    visualize_pcd_3d(original_points[:, :3], title="原始点云数据")
                
                start_time_hatch = time.time()
                # 调用粗过滤函数
                coarse_filtered_points = coarse_filtration_point_cloud(
                    original_points,
                    x_initial_threshold=config.GrabPointCalculationConfig_170.x_initial_threshold, 
                    x_stat_percentage=config.GrabPointCalculationConfig_170.x_stat_percentage,
                    x_filter_offset=config.GrabPointCalculationConfig_170.x_filter_offset,
                    intensity_threshold=config.GrabPointCalculationConfig_170.intensity_threshold, #强度
                    x_upper_bound_adjustment=config.GrabPointCalculationConfig_170.x_upper_bound_adjustment,
                    visualize_coarse_filtration=VISUALIZE_COARSE_FILTRATION)
                
                
                refined_box_corners_yz=find_rectangle_by_histogram_method(
                    coarse_filtered_points[:,:3],
                    pixel_size=config.GrabPointCalculationConfig_170.pixel_size,
                    kernel_size=config.GrabPointCalculationConfig_170.kernel_size,
                    white_threshold=config.GrabPointCalculationConfig_170.white_threshold,
                    radar_center_y=radar_center_y,
                    radar_center_z=radar_center_z,
                    is_first_time=IS_FIRST_TIME,
                    visualize_steps=VISUALIZE_RECTANGLE,)
                
                #如果船舱识别的舱口坐标为空，也就是船舱识别失败
                if refined_box_corners_yz is None:
                    if IS_FIRST_TIME :
                        raise ValueError("第一次识别失败，无法获取舱口坐标")    
                    else:
                        # 从3D坐标中提取YZ坐标，并赋值给 refined_box_corners_yz
                        # 假设X是深度，Y是水平，Z是垂直，那么YZ是第1和第2列
                        refined_box_corners_yz = hatch_corners_refined[:, 1:3]

                
                
                
                
                hatch_corners_refined, search_geometries, edge_sample_geometries = refine_x_coordinates_by_advanced_search(
                        original_points,
                        coarse_filtered_points[:,:3],
                        refined_box_corners_yz,
                        visualize_ranges=True
                    )
                x1, y1, z1 = hatch_corners_refined[0]
                x2, y2, z2 = hatch_corners_refined[1]
                x3, y3, z3 = hatch_corners_refined[2]
                x4, y4, z4 = hatch_corners_refined[3]     
                
                points_lidar = np.array([
                    [x1, y1, z1],  # 第一个点
                    [x2, y2, z2],  # 第二个点
                    [x3, y3, z3],  # 第三个点
                    [x4, y4, z4],  # 第四个点
                ])         
                
                points_world = lidar_to_world_with_x(points_lidar,translation,current_machine_position, rotation_angles)
                points_world_z=(points_world[0][2]+points_world[1][2]+points_world[2][2]+points_world[3][2])/4
                #改变世界坐标系下的z坐标
                points_world[:,2]=points_world_z
                hatch_recognize_success=True
                if not IS_FIRST_TIME and not np.allclose(points_world[:,:2], world_coords_refined[:,:2], atol=2):
                  #比较世界坐标系下的舱口坐标和上一次的舱口坐标是否合理points_world和world_coords_refined
                    if VISUALIZE_FINAL_RESULT:
                        visualize_final_result_with_search_ranges(original_points, hatch_corners_refined, search_geometries, None)

                    #logger记录points_world和world_coords_refined的值
                    logger.info(f"第{header_info.get('current_hatch', 0)}号舱的识别舱口坐标为：{points_world}")
                    logger.info(f"第{header_info.get('current_hatch', 0)}号舱的上一次识别舱口坐标为：{world_coords_refined}")


                    points_world=world_coords_refined
                    hatch_corners_refined=np.array([
                    [hatch_corners['corner1']['x'], hatch_corners['corner1']['y'], hatch_corners['corner1']['z']],
                    [hatch_corners['corner2']['x'], hatch_corners['corner2']['y'], hatch_corners['corner2']['z']],
                    [hatch_corners['corner3']['x'], hatch_corners['corner3']['y'], hatch_corners['corner3']['z']],
                    [hatch_corners['corner4']['x'], hatch_corners['corner4']['y'], hatch_corners['corner4']['z']]
                        ], dtype=np.float32)

                    hatch_recognize_success=False
                    logger.warning(f"第{header_info.get('current_hatch', 0)}号舱的识别舱口坐标与上一次差值过大，可能是识别错误")

                  
                print("世界坐标系下的点：")
                for i, (x, y, z) in enumerate(points_world):
                    print(f"点{i + 1}: ({x:+.3f}, {y:+.3f}, {z:+.3f})")

                if VISUALIZE_FINAL_RESULT and hatch_recognize_success:
                    visualize_final_result_with_search_ranges(original_points, hatch_corners_refined, search_geometries, None)
                
                end_time_hatch = time.time()
                hatch_time = end_time_hatch - start_time_hatch
                logger.info(f"舱口识别时间为：{hatch_time}")
                
                
                
                
                
                
                start_time_coal = time.time()
                original_world_points = lidar_to_world_with_x(original_points[:, :3], translation, current_machine_position, rotation_angles)

                world_coal_pile_points = segment_coal_pile_world_points(
                    original_world_points, points_world,
                    eps_coal = config.GrabPointCalculationConfig_170.eps_coal,
                    min_samples_coal = config.GrabPointCalculationConfig_170.min_samples_coal,
                    z_changes=config.GrabPointCalculationConfig_170.z_changes,
                    xy_shrink_amount=config.GrabPointCalculationConfig_170.xy_shrink_amount,
                    visualize_dbscan=visualize_clustered_pcd
                        )

                if visualize_coal_points:
                    #可视化世界坐标系下的煤堆点
                    visualize_pcd_3d(world_coal_pile_points, title="世界坐标系下的煤堆点")

                if len(world_coal_pile_points) > 0:
                    # print(f"成功分割出煤堆，包含 {len(coal_pile_points)} 个点")
                     # 构造煤堆数据
                    coal_pile_data = {
                        "frame_id": header_info.get('frame_id', 0),
                        "current_hatch": header_info.get('current_hatch', 0),
                        "point_count": len(world_coal_pile_points),
                        "detection_success": len(world_coal_pile_points) > 0,
                        "hatch_corners": {
                            "corner1": {"x": float(hatch_corners_refined[0][0]), "y": float(hatch_corners_refined[0][1]), "z": float(hatch_corners_refined[0][2])},
                            "corner2": {"x": float(hatch_corners_refined[1][0]), "y": float(hatch_corners_refined[1][1]), "z": float(hatch_corners_refined[1][2])},
                            "corner3": {"x": float(hatch_corners_refined[2][0]), "y": float(hatch_corners_refined[2][1]), "z": float(hatch_corners_refined[2][2])},
                            "corner4": {"x": float(hatch_corners_refined[3][0]), "y": float(hatch_corners_refined[3][1]), "z": float(hatch_corners_refined[3][2])}
                        },
                         "world_corners": {
                            "corner1": {"x": float(points_world[0][0]), "y": float(points_world[0][1]), "z": float(points_world[0][2])},
                            "corner2": {"x": float(points_world[1][0]), "y": float(points_world[1][1]), "z": float(points_world[1][2])},
                            "corner3": {"x": float(points_world[2][0]), "y": float(points_world[2][1]), "z": float(points_world[2][2])},
                            "corner4": {"x": float(points_world[3][0]), "y": float(points_world[3][1]), "z": float(points_world[3][2])}
                        }
                    }
                    # 添加煤堆点云数据
                    points_list = []
                    for point in world_coal_pile_points:
                        points_list.append({
                            "x": float(point[0]),
                            "y": float(point[1]),
                            "z": float(point[2]),
                            #"intensity": float(point[3]) if len(point) > 3 else 0.0
                        })
                    coal_pile_data["coal_pile_points"] = points_list
                    

                else:

                    logger.info("未检测到煤堆")
                
                end_time_coal = time.time()
                coal_time = end_time_coal - start_time_coal
                logger.info(f"煤堆分割时间为：{coal_time}")
                
                
                
                
                
                
                
                start_time_grab=time.time()
                # line_width=config.GrabPointCalculationConfig_170.line_width  #线宽
                # floor_height=config.GrabPointCalculationConfig_170.floor_height  #层高
                # safe_distance_x_negative_init=config.GrabPointCalculationConfig_170.safe_distance_x_negative_init  #大车左侧安全距离
                # safe_distance_x_positive_init=config.GrabPointCalculationConfig_170.safe_distance_x_positive_init  #大车右侧安全距离
                # safe_distance_y_ocean_init=config.GrabPointCalculationConfig_170.safe_distance_y_ocean_init  #海侧安全距离
                # safe_distance_y_land_init=config.GrabPointCalculationConfig_170.safe_distance_y_land_init  #陆侧安全距离
                # hatch_depth=config.GrabPointCalculationConfig_170.hatch_depth  # 型深
                # line_gap=config.GrabPointCalculationConfig_170.line_gap  #线和线之间的间隔
                # expansion_x_front=config.GrabPointCalculationConfig_170.expansion_x_front  #前四层大车方向（x）外扩系数.
                # expansion_y_front=config.GrabPointCalculationConfig_170.expansion_y_front  #前四层小车方向（y）外扩系数.
                
                # expansion_x_back=config.GrabPointCalculationConfig_170.expansion_x_back  #后四层大车方向（x）外扩系数.
                # expansion_y_back=config.GrabPointCalculationConfig_170.expansion_y_back  #后四层小车方向（y）外扩系数.
                # block_width=config.GrabPointCalculationConfig_170.block_width  #每个分块的宽度
                # block_length=config.GrabPointCalculationConfig_170.block_length  #每个分块的长度    
                # plane_threshold=config.GrabPointCalculationConfig_170.plane_threshold  #平面阈值
                # plane_distance=config.GrabPointCalculationConfig_170.plane_distance  #平面的情况抓取点移动的距离
                # bevel_distance=config.GrabPointCalculationConfig_170.bevel_distance  #斜面的情况抓取点移动的距离
                # retain_height=config.GrabPointCalculationConfig_170.retain_height  #保留高度
                # Sign=config.GrabPointCalculationConfig_170.Sign  #使用线性函数
                # k=config.GrabPointCalculationConfig_170.k  #线性函数的斜率
                # b=config.GrabPointCalculationConfig_170.b  #线性函数的截距

                line_width=header_info.get('lineWidth',4)  #线宽
                floor_height=header_info.get('floorHeight',2)  #层高
                safe_distance_x_negative_init=header_info.get('safeDistanceXNegativeInit',5)  #大车左侧安全距离
                safe_distance_x_positive_init=header_info.get('safeDistanceXPositiveInit',5)  #大车右侧安全距离
                safe_distance_y_ocean_init=header_info.get('safeDistanceYOceanInit',5)  #海侧安全距离
                safe_distance_y_land_init=header_info.get('safeDistanceYLandInit',5)  #陆侧安全距离
                hatch_depth=header_info.get('hatchDepth',16)  # 型深
                line_gap=header_info.get('lineGap',0.5)  #线和线之间的间隔
                expansion_x_front=header_info.get('expansionXFront',0.4)  #前四层大车方向（x）外扩系数.
                expansion_y_front=header_info.get('expansionYFront',0.5)  #前四层小车方向（y）外扩系数.
                expansion_x_back=header_info.get('expansionXBack',0.9)  #后四层大车方向（x）外扩系数.
                expansion_y_back=header_info.get('expansionYBack',0.9)  #后四层小车方向（y）外扩系数.
                block_width=header_info.get('blockWidth',1)  #每个分块的宽度
                block_length=header_info.get('blockLength',1)  #每个分块的长度    
                plane_threshold=header_info.get('planeThreshold',0.3)  #平面阈值
                plane_distance=header_info.get('planeDistance',1)  #平面的情况抓取点移动的距离
                bevel_distance=header_info.get('bevelDistance',2.5)  #斜面的情况抓取点移动的距离
                retain_height=header_info.get('retainHeight',0.5)  #保留高度
                Sign=header_info.get('signFlag',0)  #使用线性函数
                k=header_info.get('kValue',1)  #线性函数的斜率
                b=header_info.get('bValue',-1)  #线性函数的截距
                config_height_diff=header_info.get('heightDiff',3.5)  #高度差
                selected_line=header_info.get('selected_line',0)  #选中的线



                
                #最大高度差
                max_height_diff=hatch_depth-retain_height
                #当前煤面的高度,即所有煤堆点的z坐标平均值
                current_coal_height=world_coal_pile_points[:,2].mean()
                #当前舱口高度
                hatch_height=(points_world[0][2]+points_world[1][2]+points_world[2][2]+points_world[3][2])/4
                
                #计算当前煤面的高度与舱口高度的差值
                height_diff=abs(current_coal_height-hatch_height)
                logger.info(f"当前煤面的高度为：{current_coal_height}")
                logger.info(f"舱口的高度为：{hatch_height}")
                logger.info(f"当前煤面的高度与舱口高度的差值为：{height_diff}")







                #计算当前煤面在哪一层
                current_layer = math.ceil(height_diff / floor_height)
                logger.info(f"当前煤面在第{current_layer}层")
                #计算安全边界
                if current_layer<=4:
                    safe_distance_x_negative=safe_distance_x_negative_init-((current_layer-1)*expansion_x_front)
                    safe_distance_x_positive=safe_distance_x_positive_init-((current_layer-1)*expansion_x_front)
                    safe_distance_y_ocean=safe_distance_y_ocean_init-((current_layer-1)*expansion_y_front)
                    safe_distance_y_land=safe_distance_y_land_init-((current_layer-1)*expansion_y_front)
                    if safe_distance_x_negative<=0.2:
                        safe_distance_x_negative=0.2
                    if safe_distance_x_positive<=0.2:
                        safe_distance_x_positive=0.2
                    if safe_distance_y_ocean<=0:
                        safe_distance_y_ocean=0
                    if safe_distance_y_land<=0:
                        safe_distance_y_land=0
                
                    logger.info(f"大车左侧安全距离为：{safe_distance_x_negative}")
                    logger.info(f"大车右侧安全距离为：{safe_distance_x_positive}")
                    logger.info(f"海侧安全距离为：{safe_distance_y_ocean}")
                    logger.info(f"陆侧安全距离为：{safe_distance_y_land}")
                else:
                    safe_distance_x_negative=safe_distance_x_negative_init-(3*expansion_x_front)-((current_layer-4)*expansion_x_back)
                    safe_distance_x_positive=safe_distance_x_positive_init-(3*expansion_x_front)-((current_layer-4)*expansion_x_back)
                    safe_distance_y_ocean=safe_distance_y_ocean_init-(3*expansion_y_front)-((current_layer-4)*expansion_y_back)
                    safe_distance_y_land=safe_distance_y_land_init-(3*expansion_y_front)-((current_layer-4)*expansion_y_back)
                
                    if safe_distance_x_negative<=0.2:
                        safe_distance_x_negative=0.2
                    if safe_distance_x_positive<=0.2:
                        safe_distance_x_positive=0.2
                    if safe_distance_y_ocean<=0:
                        safe_distance_y_ocean=0
                    if safe_distance_y_land<=0:
                        safe_distance_y_land=0
                    logger.info(f"大车左侧安全距离为：{safe_distance_x_negative}")
                    logger.info(f"大车右侧安全距离为：{safe_distance_x_positive}")
                    logger.info(f"海侧安全距离为：{safe_distance_y_ocean}")
                    logger.info(f"陆侧安全距离为：{safe_distance_y_land}")
                
                #计算舱口的长宽
                hatch_width=abs(points_world[1][1]-points_world[0][1])
                hatch_length=abs(points_world[2][0]-points_world[1][0])
                logger.info(f"舱口的宽度为：{hatch_width}")
                logger.info(f"舱口的长度为：{hatch_length}")
                
                #确定线的位置
                #舱口长宽减去安全距离的xy坐标轴范围
                
                x_left=max(points_world[1][0],points_world[0][0])
                x_right=min(points_world[2][0],points_world[3][0])

                x_negative =max(points_world[1][0],points_world[0][0])+safe_distance_x_negative
                x_positive=min(points_world[2][0],points_world[3][0])-safe_distance_x_positive
                
                y_front=min(points_world[1][1],points_world[2][1])
                y_back=max(points_world[0][1],points_world[3][1])
                if current_layer<config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer:
                    y_grab_expansion=config.GrabPointCalculationConfig_170.y_grab_expansion
                if config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer<=current_layer<config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer2:
                    y_grab_expansion=config.GrabPointCalculationConfig_170.y_grab_expansion2

                if current_layer>=config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer2:
                    y_grab_expansion=config.GrabPointCalculationConfig_170.y_grab_expansion3
                y_ocean=min(points_world[1][1],points_world[2][1])-safe_distance_y_ocean+y_grab_expansion
                y_land=max(points_world[0][1],points_world[3][1])+safe_distance_y_land-y_grab_expansion
                logger.info(f"x_positive为：{x_positive}")
                logger.info(f"x_negative为：{x_negative}")
                logger.info(f"y_ocean为：{y_ocean}")
                logger.info(f"y_land为：{y_land}")
                
                # #第一条线的位置
                
                # line1_x_negative=current_machine_position-line_width/2
                # line1_x_positive=current_machine_position+line_width/2
                
                # lines= [[line1_x_negative,line1_x_positive]]
                
                # #向x轴负方向生成线
                # negative_edge=line1_x_negative
                # while True:
                #   next_negative_edge=negative_edge-(line_width+line_gap)
                #   if next_negative_edge>=x_negative:
                #     lines.append([next_negative_edge,next_negative_edge+line_width])
                #     negative_edge=next_negative_edge
                #   else:
                #     break
                
                # #向x轴正方向生成线
                # positive_edge=line1_x_positive
                # while True:
                #   next_positive_edge=positive_edge+(line_width+line_gap)
                #   if next_positive_edge<=x_positive:
                #     lines.append([next_positive_edge-line_width,next_positive_edge])
                #     positive_edge=next_positive_edge
                #   else:
                #     break
                
                #现在先以x_negative和x_positive为线的中心基准，生成两条线
                square_ranges=[]
                nine_square_block_heights=[]

                nine_squarecalculate(
                        square_ranges=square_ranges,
                        nine_square_block_heights=nine_square_block_heights,
                        world_coal_pile_points=world_coal_pile_points,
                        x_negative=x_negative-2,
                        x_positive=x_positive+2,
                        y_ocean=y_ocean,
                        y_land=y_land,
                        block_width=block_width,
                        block_length=block_length,
                    )



                line1_x_negative=x_negative-line_width/2
                line1_x_positive=x_negative+line_width/2
                line2_x_negative=x_positive-line_width/2
                line2_x_positive=x_positive+line_width/2
                lines=[[line1_x_negative,line1_x_positive],[line2_x_negative,line2_x_positive]]
                
                #剩下的x轴范围,两边都预留了半个line_gap的宽度
                x_remaining=line2_x_negative-line1_x_positive-line_gap
                
                #计算可以生成多少条线
                num_lines=int(x_remaining/(line_width+line_gap))
                #所以现在每条线的宽度为
                line_w=x_remaining/num_lines
                
                for i in range(num_lines):
                    #线的中心点位置
                    line_center=(line1_x_positive+line_gap/2)+i*line_w+0.5*line_w
                
                    lines.append([line_center-0.5*line_width,line_center+0.5*line_width])
                
                
                #给这些线编号，从负方向开始
                lines_sorted = sorted(lines, key=lambda l: (l[0] + l[1]) / 2)


                lines_dict_tojava = {idx + 1: line for idx, line in enumerate(lines_sorted)}
                if mode_flag==4:
                    
                    
                    lines_dict={}

                    lines_dict={1:(square_ranges[0]["x_min"],square_ranges[0]["x_max"]),2:(square_ranges[3]["x_min"],square_ranges[3]["x_max"]),3:(square_ranges[6]["x_min"],square_ranges[6]["x_max"])}


                else:
                    # 给每条线编号（从负方向到正方向）
                    lines_dict = {idx + 1: line for idx, line in enumerate(lines_sorted)}
                    for num, (x_min, x_max) in lines_dict.items():
                        logger.info(f"编号 {num}：范围=({x_min:.2f}, {x_max:.2f})")
                
                
                #定义一个字典，用来保存每条线的高度
                line_heights_dict={}

                #提取每条线内的煤堆点，不仅有x的限制还有y方向的限制，计算每条线内的点的z坐标的平均值，作为这条线的高度
                for num, (x_min, x_max) in lines_dict.items():
                    # 提取在当前线框内的点
                    line_points = world_coal_pile_points[(world_coal_pile_points[:, 0] >= x_min) & (world_coal_pile_points[:, 0] <= x_max)]
                    # 提取在当前线框内且y坐标在安全范围内的点
                    line_points = line_points[(line_points[:, 1] <= y_ocean) & (line_points[:, 1] >= y_land)]
                    #计算这条线的高度
                    line_height=line_points[:,2].mean()
                    logger.info(f"编号 {num} 线内的煤堆点的平均高度: {line_height}")
                    #保存在一个字典内
                    line_heights_dict[num]=line_height

                current_line=None

                if last_capture_point_x!=0:
                    #判断上次抓取的点在哪条线上

                    for num, (x_min, x_max) in lines_dict.items():
                        if last_capture_point_x>=x_min and last_capture_point_x<=x_max:
                            current_line=num
                            break
                    
                else:
                    current_line=None
                    

                selected_map=[]
                if mode_flag==4:
                    work_completed=False
                    #选中的格子,从header_info["selection_blocks"]中提取值为1的格子的索引
                    selected_map=[index for index,value in enumerate(header_info["selection_blocks"]) if value==1]
                    logger.info(f"选中的格子为：{selected_map}")
                    if len(selected_map)==0:
                        logger.error("没有选中任何格子")
                        raise ValueError("没有选中任何格子")

                    #有格子的线
                    lines_with_map=[]
                    #遍历selected_map判断这些格子分别处于哪条线
                    for index in selected_map:
                        line=index//3+1
                        if line not in lines_with_map:
                            lines_with_map.append(line)

                    if current_line in lines_with_map:
                        logger.info(f"上次抓取的点在第{current_line}条线上")
                    else:
                        current_line=None

                    # current_line=None
                    # for num, (x_min, x_max) in lines_dict.items():
                    #     if current_machine_position>=x_min and current_machine_position<=x_max and num in lines_with_map:
                    #         current_line=num
                    #         break

                    if current_line is None:
                        
                        logger.info("大车当前位置不在任何一条可以作业的线上，将去往平均高度最高的那条线且可以作业的线")
                        #获取平均高度最高的那条线且在lines_with_map中的线
                        current_line=max([line for line in line_heights_dict.keys() if line in lines_with_map], key=line_heights_dict.get)
                        logger.info(f"平均高度最高的那条线为：{current_line}，即将前往{current_line}线")
                        # #跳过这次循环
                        # continue.
                
                    else:
                        logger.info(f"大车当前处于第{current_line}条线")
                
                
                    #计算当前大车所在线的高度和其他所有线的差值，如果有一个差值大于3.5米的话，且当前线所在高度是相减的两者之间较小的话，就启动换线.或者当前线的平均高度已经到了保留的高度，也启动换线
                    current_line_height=line_heights_dict[current_line]
                    #是否需要换线
                    need_change_line=False

                    for num, height in line_heights_dict.items():
                        if num!=current_line:
                            height_diff=abs(current_line_height-height)
                            if (height_diff>config_height_diff and current_line_height<height) or ((hatch_height-current_line_height)>=max_height_diff):
                                if (height_diff>config_height_diff and current_line_height<height):
                                    logger.info(f"当前大车所在线的高度为：{current_line_height}，第{num}条线的高度为：{height}，差值为：{height_diff}，大于{config_height_diff}米，需要换线")
                                if (hatch_height-current_line_height)>=max_height_diff:
                                    logger.info(f"当前大车所在线的高度为：{current_line_height}，低于保留高度，需要换线")
                                #启动换线
                                need_change_line=True

                    
                                break
                                    
                    if len(lines_with_map)==1 and need_change_line:
                        work_completed=True
                        need_change_line=False
                   
                    if need_change_line:
                  # 如果需要换线，就循环找下一条合适的线
                     # 记录已经尝试过的线，防止无限循环
                        tried_lines = set([current_line])
                        total_lines = len(lines_with_map)
                        while need_change_line:
                            next_line, persisted_line_change_direction = get_next_line(
                                current_line,
                                persisted_line_change_direction,
                                lines_with_map,
                                tried_lines
                            )
                            logger.info(f"换线到第{next_line}条线，当前线的边界位置为：{lines_dict[next_line]}")
                            tried_lines.add(next_line)
                            current_line = next_line
                            current_line_height = line_heights_dict[current_line]
                        
                        # 再次判断新线是否需要换线
                            need_change_line = False
                            for num, height in line_heights_dict.items():
                                if num != current_line:
                                    height_diff = abs(current_line_height - height)
                                    if (height_diff>config_height_diff and current_line_height<height) or ((hatch_height-current_line_height)>=max_height_diff):
                                        if (height_diff>config_height_diff and current_line_height<height):
                                            logger.info(f"当前大车所在线的高度为：{current_line_height}，第{num}条线的高度为：{height}，差值为：{height_diff}，大于{config_height_diff}米，需要换线")
                                        if (hatch_height-current_line_height)>=max_height_diff:
                                            logger.info(f"当前大车所在线的高度为：{current_line_height}，低于保留高度，需要换线")
                                        need_change_line = True
                                        break
                            if len(tried_lines)==total_lines and need_change_line:
                                work_completed=True
                                raise Exception("换线失败，已经尝试过所有的线，作业结束")  
                                break      


                else:
                #根据大车当前位置，确定大车当前处于哪一条线
                    # current_line=None
                    # for num, (x_min, x_max) in lines_dict.items():
                    #     if current_machine_position>=x_min and current_machine_position<=x_max:
                    #         current_line=num
                    #         break

                    
                    if current_line is None:
                        
                    
                        logger.info("大车当前位置不在任何一条线上，将去往平均高度最高的那条线")
                        #获取平均高度最高的那条线
                        current_line=max(line_heights_dict, key=line_heights_dict.get)
                        logger.info(f"平均高度最高的那条线为：{current_line}，即将前往{current_line}线")
                        # #跳过这次循环
                        # continue.
                
                    else:
                        logger.info(f"大车当前处于第{current_line}条线")

                    if selected_line!=0 and selected_line in lines_dict.keys():
                        current_line=selected_line
                        logger.info(f"用户选择了第{selected_line}条线，即将前往{selected_line}线")
                    
                    #计算当前大车所在线的高度和其他所有线的差值，如果有一个差值大于3.5米的话，且当前线所在高度是相减的两者之间较小的话，就启动换线.或者当前线的平均高度已经到了保留的高度，也启动换线
                    current_line_height=line_heights_dict[current_line]
                    #是否需要换线
                    need_change_line=False

                    for num, height in line_heights_dict.items():
                        if num!=current_line:
                            height_diff=abs(current_line_height-height)
                            if (height_diff>config_height_diff and current_line_height<height) or ((hatch_height-current_line_height)>=max_height_diff):
                                if (height_diff>config_height_diff and current_line_height<height):
                                    logger.info(f"当前大车所在线的高度为：{current_line_height}，第{num}条线的高度为：{height}，差值为：{height_diff}，大于{config_height_diff}米，需要换线")
                                if (hatch_height-current_line_height)>=max_height_diff:
                                    logger.info(f"当前大车所在线的高度为：{current_line_height}，低于保留高度，需要换线")
                                #启动换线
                                need_change_line=True

                    
                                break
                                    

                    work_completed=False
                    if need_change_line:
                  # 如果需要换线，就循环找下一条合适的线
                     # 记录已经尝试过的线，防止无限循环
                        tried_lines = set([current_line])
                        total_lines = len(lines_dict)
                        while need_change_line:
                            next_line, persisted_line_change_direction = get_next_line(
                                current_line,
                                persisted_line_change_direction,
                                list(lines_dict.keys()),
                                tried_lines
                            )
                            logger.info(f"换线到第{next_line}条线，当前线的边界位置为：{lines_dict[next_line]}")
                            tried_lines.add(next_line)
                        # 更新当前线
                            current_line = next_line
                            current_line_height = line_heights_dict[current_line]
                        
                        # 再次判断新线是否需要换线
                            need_change_line = False
                            for num, height in line_heights_dict.items():
                                if num != current_line:
                                    height_diff = abs(current_line_height - height)
                                    if (height_diff>config_height_diff and current_line_height<height) or ((hatch_height-current_line_height)>=max_height_diff):
                                        if (height_diff>config_height_diff and current_line_height<height):
                                            logger.info(f"当前大车所在线的高度为：{current_line_height}，第{num}条线的高度为：{height}，差值为：{height_diff}，大于{config_height_diff}米，需要换线")
                                        if (hatch_height-current_line_height)>=max_height_diff:
                                            logger.info(f"当前大车所在线的高度为：{current_line_height}，低于保留高度，需要换线")
                                        need_change_line = True
                                        break
                            if len(tried_lines)==total_lines and need_change_line:
                                work_completed=True
                            #异步启动船舱深度测量，不阻塞后续计算
                                try:
                                    # 保存煤堆点云数据到临时文件，使用进程ID区分
                                    process_id = "170"
                                    temp_data_file = os.path.join(os.path.dirname(__file__), f"temp_coal_pile_data_{process_id}.npy")
                                    np.save(temp_data_file, world_coal_pile_points)
                                    logger.info(f"已保存煤堆点云数据到临时文件: {temp_data_file}")
                                    
                                    # 启动 measure_depth.py脚本，传递数据文件路径、舱口号和进程ID
                                    script_path = os.path.join(os.path.dirname(__file__), "measure_depth.py")
                                    subprocess.Popen([sys.executable, script_path, temp_data_file, str(current_hatch), str(hatch_height), process_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    logger.info(f"已异步启动船舱深度测量任务 measure_depth.py，舱口号: {current_hatch}，进程ID: {process_id}")
                                    # 在需要使用舱底深度的地方

                                except Exception as e:
                                    logger.exception(f"异步启动船舱深度测量失败: {e}",exc_info=True)

                                #抛出一个指定的异常
                                raise Exception("换线失败，已经尝试过所有的线，作业结束")

                                break

                                        
                logger.info(f"当前线为第{current_line}条线，当前线的边界位置为：{lines_dict[current_line]}")
                
                # capture_point2={'x':0.0,'y':0.0,'z':0.0}
                # capture_point2_layer=0
                
                
                # 计算当前线在哪一层
                current_line_layer = math.ceil(abs(hatch_height - current_line_height) / floor_height)

                if current_line_layer!=current_layer:
                    if current_line_layer<=4:
                        safe_distance_y_ocean=safe_distance_y_ocean_init-((current_line_layer-1)*expansion_y_front)
                        safe_distance_y_land=safe_distance_y_land_init-((current_line_layer-1)*expansion_y_front)
                        if safe_distance_y_ocean<=0:
                            safe_distance_y_ocean=0
                        if safe_distance_y_land<=0:
                            safe_distance_y_land=0
                    
                        logger.info(f"海侧安全距离为：{safe_distance_y_ocean}")
                        logger.info(f"陆侧安全距离为：{safe_distance_y_land}")
                    else:
                        safe_distance_y_ocean=safe_distance_y_ocean_init-(3*expansion_y_front)-((current_line_layer-4)*expansion_y_back)
                        safe_distance_y_land=safe_distance_y_land_init-(3*expansion_y_front)-((current_line_layer-4)*expansion_y_back)
                    
                        if safe_distance_y_ocean<=0:
                            safe_distance_y_ocean=0
                        if safe_distance_y_land<=0:
                            safe_distance_y_land=0
                        logger.info(f"海侧安全距离为：{safe_distance_y_ocean}")
                        logger.info(f"陆侧安全距离为：{safe_distance_y_land}")




                if current_layer<config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer:
                    y_grab_expansion=config.GrabPointCalculationConfig_170.y_grab_expansion
                if config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer<=current_layer<config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer2:
                    y_grab_expansion=config.GrabPointCalculationConfig_170.y_grab_expansion2

                if current_layer>=config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer2:
                    y_grab_expansion=config.GrabPointCalculationConfig_170.y_grab_expansion3

                    y_ocean=min(points_world[1][1],points_world[2][1])-safe_distance_y_ocean+y_grab_expansion
                    y_land=max(points_world[0][1],points_world[3][1])+safe_distance_y_land-y_grab_expansion


                    square_ranges=[]
                    nine_square_block_heights=[]
                    nine_squarecalculate(
                    square_ranges=square_ranges,
                    nine_square_block_heights=nine_square_block_heights,
                    world_coal_pile_points=world_coal_pile_points,
                    x_negative=x_negative-2,
                    x_positive=x_positive+2,
                    y_ocean=y_ocean,
                    y_land=y_land,
                    block_width=block_width,
                    block_length=block_length,
                      )




                limited_layers=config.GrabPointCalculationConfig_170.limited_layers
                limited_height=hatch_height-(config.GrabPointCalculationConfig_170.limited_height*floor_height)
                x_dump_truck=config.GrabPointCalculationConfig_170.x_dump_truck
                y_dump_truck=config.GrabPointCalculationConfig_170.y_dump_truck
                limited_change_height_land=config.GrabPointCalculationConfig_170.limited_change_height_land
                limited_change_height_ocean=config.GrabPointCalculationConfig_170.limited_change_height_ocean
                limited_change_height_land_x_dump=config.GrabPointCalculationConfig_170.limited_change_height_land_x_dump
                limited_change_height_ocean_x_dump=config.GrabPointCalculationConfig_170.limited_change_height_ocean_x_dump
                limited_change_height_normal_x_dump=config.GrabPointCalculationConfig_170.limited_change_height_normal_x_dump
                
                limited_layers_y_dump_truck=config.GrabPointCalculationConfig_170.limited_layers_y_dump_truck
                limited_layers_x_dump_truck=config.GrabPointCalculationConfig_170.limited_layers_x_dump_truck
                max_height_var_change=config.GrabPointCalculationConfig_170.max_height_var_change
                land_to_centerline=config.GrabPointCalculationConfig_170.land_to_centerline
                ocean_to_centerline=config.GrabPointCalculationConfig_170.ocean_to_centerline
                y_offset_layer=config.GrabPointCalculationConfig_170.y_offset_layer
                y_offset_land=config.GrabPointCalculationConfig_170.y_offset_land
                y_offset_ocean=config.GrabPointCalculationConfig_170.y_offset_ocean
                y_offset_land_xy_dump=config.GrabPointCalculationConfig_170.y_offset_land_xy_dump
                y_offset_ocean_xy_dump=config.GrabPointCalculationConfig_170.y_offset_ocean_xy_dump



                
                if current_line_layer>=limited_layers:
                    enable_limited_flag=True
                else:
                    enable_limited_flag=False


                if current_line_layer>=y_offset_layer:
                    enable_y_offset_flag=True
                else:
                    enable_y_offset_flag=False

                if current_line_layer>=config.GrabPointCalculationConfig_170.y_grab_expansion_change_layer:
                    y_grab_expansion_change_flag=True
                else:
                    y_grab_expansion_change_flag=False
                #大车方向甩斗的高度限制

                above_current_line_layer=current_line_layer-limited_layers_x_dump_truck
                above_current_line_layer_min_height=hatch_height-(above_current_line_layer*floor_height)


                #小车方向甩斗的高度限制
                above_current_line_layer2=current_line_layer-limited_layers_y_dump_truck
                above_current_line_layer2_min_height=hatch_height-(above_current_line_layer2*floor_height)
                blocks_heights={}
                # process_blocks_for_line(
                #     blocks_heights=blocks_heights,
                #     world_coal_pile_points=world_coal_pile_points,
                #     lines_dict=lines_dict,
                #     y_land=y_land,
                #     y_ocean=y_ocean,
                #     block_width=block_width,
                #     block_length=block_length,
                #     enable_limited_flag=enable_limited_flag,
                #     logger=logger
                # )


                need_calculate_two=False
                if last_capture_point_x==0 and last_capture_point_y==0 and last_capture_point_z==0:
                    need_calculate_two=True
                exclude_radius=1.2

                # 动态递减屏蔽半径重试封装
                def calculate_capture_point_with_backoff(exclude_x_center, exclude_y_center):
                    radius = exclude_radius
                    min_radius = 0.2
                    step = 0.2
                    last_error = None
                    while radius >= min_radius:
                        try:
                            return calculate_capture_point(
                                world_coal_pile_points=world_coal_pile_points,
                                lines_dict=lines_dict,
                                current_line=current_line,
                                exclude_x_center=exclude_x_center,
                                exclude_y_center=exclude_y_center,
                                exclude_radius=radius,
                                x_left=x_left,
                                x_right=x_right,
                                y_front=y_front,
                                y_back=y_back,
                                y_land=y_land,
                                y_ocean=y_ocean,
                                hatch_height=hatch_height,
                                current_line_height=current_line_height,
                                floor_height=floor_height,
                                block_width=block_width,
                                block_length=block_length,
                                line_width=line_width,
                                plane_threshold=plane_threshold,
                                plane_distance=plane_distance,
                                bevel_distance=bevel_distance,
                                Sign=Sign,
                                k=k,
                                b=b,
                                above_current_line_layer_min_height=above_current_line_layer_min_height,
                                enable_limited_flag=enable_limited_flag,
                                limited_height=limited_height,
                                x_dump_truck=x_dump_truck,
                                y_dump_truck=y_dump_truck,
                                limited_change_height_land=limited_change_height_land,
                                limited_change_height_ocean=limited_change_height_ocean,
                                limited_change_height_land_x_dump=limited_change_height_land_x_dump,
                                limited_change_height_ocean_x_dump=limited_change_height_ocean_x_dump,
                                limited_change_height_normal_x_dump=limited_change_height_normal_x_dump,
                                above_current_line_layer2_min_height=above_current_line_layer2_min_height,
                                mode_flag=mode_flag,
                                land_to_centerline=land_to_centerline,
                                ocean_to_centerline=ocean_to_centerline,
                                logger=logger,
                                max_height_var_change =max_height_var_change,
                                y_offset_land=y_offset_land,
                                y_offset_ocean=y_offset_ocean,
                                y_offset_land_xy_dump=y_offset_land_xy_dump,
                                y_offset_ocean_xy_dump=y_offset_ocean_xy_dump,
                                enable_y_offset_flag=enable_y_offset_flag,
                                y_grab_expansion_change_flag=y_grab_expansion_change_flag,
                                square_ranges=square_ranges,
                                selected_map=selected_map
                            )
                        except Exception as e:
                            last_error = e
                            logger.warning(f"抓取点计算失败，屏蔽半径 {radius:.2f}，递减重试: {e}",exc_info=True)
                            radius -= step
                    logger.warning("尝试关闭屏蔽区域以获取抓取点")
                    return calculate_capture_point(
                        world_coal_pile_points=world_coal_pile_points,
                        lines_dict=lines_dict,
                        current_line=current_line,
                        exclude_x_center=exclude_x_center,
                        exclude_y_center=exclude_y_center,
                        exclude_radius=0.0,
                        x_left=x_left,
                        x_right=x_right,
                        y_front=y_front,
                        y_back=y_back,
                        y_land=y_land,
                        y_ocean=y_ocean,
                        hatch_height=hatch_height,
                        current_line_height=current_line_height,
                        floor_height=floor_height,
                        block_width=block_width,
                        block_length=block_length,
                        line_width=line_width,
                        plane_threshold=plane_threshold,
                        plane_distance=plane_distance,
                        bevel_distance=bevel_distance,
                        Sign=Sign,
                        k=k,
                        b=b,
                        above_current_line_layer_min_height=above_current_line_layer_min_height,
                        enable_limited_flag=enable_limited_flag,
                        limited_height=limited_height,
                        x_dump_truck=x_dump_truck,
                        y_dump_truck=y_dump_truck,
                        limited_change_height_land=limited_change_height_land,
                        limited_change_height_ocean=limited_change_height_ocean,
                        limited_change_height_land_x_dump=limited_change_height_land_x_dump,
                        limited_change_height_ocean_x_dump=limited_change_height_ocean_x_dump,
                        limited_change_height_normal_x_dump=limited_change_height_normal_x_dump,
                        above_current_line_layer2_min_height=above_current_line_layer2_min_height,
                        mode_flag=mode_flag,
                        land_to_centerline=land_to_centerline,
                        ocean_to_centerline=ocean_to_centerline,
                        logger=logger,
                        max_height_var_change =max_height_var_change,
                        y_offset_land=y_offset_land,
                        y_offset_ocean=y_offset_ocean,
                        y_offset_land_xy_dump=y_offset_land_xy_dump,
                        y_offset_ocean_xy_dump=y_offset_ocean_xy_dump,
                        enable_y_offset_flag=enable_y_offset_flag,
                        y_grab_expansion_change_flag=y_grab_expansion_change_flag,
                        square_ranges=square_ranges,
                        selected_map=selected_map
                    )

                if need_calculate_two:
                    # 计算第一个抓取点（递减屏蔽半径）
                    capture_point, capture_point_layer = calculate_capture_point_with_backoff(
                        last_capture_point_x,
                        last_capture_point_y
                    )
                    capture_point_layer_min_height=hatch_height-(capture_point_layer*floor_height)
                    capture_point_effective=1
                    # 计算第二个抓取点，排除第一个抓取点的区域（递减屏蔽半径）
                    capture_point2, capture_point2_layer = calculate_capture_point_with_backoff(
                        capture_point['x'],
                        capture_point['y']
                    )
                    capture_point2_layer_min_height=hatch_height-(capture_point2_layer*floor_height)

                else:
                    capture_point={'x':last_capture_point_x,'y':last_capture_point_y,'z':last_capture_point_z}
                    capture_point_layer=0
                    capture_point_effective=0
                    capture_point_layer_min_height=0
                    # 有上次点时的抓取（递减屏蔽半径）
                    capture_point2,capture_point2_layer=calculate_capture_point_with_backoff(
                        last_capture_point_x,
                        last_capture_point_y
                    )
                    capture_point2_layer_min_height=hatch_height-(capture_point2_layer*floor_height)
                    # if abs(capture_point2['x']-capture_point['x'])<0.5:
                    #   logger.info(f'使用上次的抓取点的x坐标{capture_point["x"]}')
                    #   capture_point2['x']=capture_point['x']
                        
                
                # print(f'blocks_heights={blocks_heights}')
                #将blocks_heights转换为二维列表，原来字典的键变为索引ij
                blocks_heights_list = []
                # 获取所有的i和j值
                i_values = sorted(set([key[0] for key in blocks_heights.keys()]))
                j_values = sorted(set([key[1] for key in blocks_heights.keys()]))
                
                # 创建二维列表
                for i in i_values:
                    row = []
                    for j in j_values:
                        if (i, j) in blocks_heights:
                            row.append(blocks_heights[(i, j)])
                        else:
                            row.append(None)  # 如果没有对应的键，填充None
                    blocks_heights_list.append(row)
                
                
                
                
                # #计算本线所处的那一层的最低点
                # current_line_layer_min_height=hatch_height-(current_line_layer*floor_height)
                
                end_time_grab=time.time()
                logger.info(f"计算时间为：{end_time_grab-start_time_grab}")                
                


                
                
                

                
            except Exception as e:
                # 记录异常但不退出循环
                

                
                if "换线失败" in str(e):
                    logger.error(f"换线失败，已经尝试过所有的线，作业结束")
                    #发给java后台的数据,，type=4 表示作业结束，没有可以作业的线
                    to_java_data={
                        'type':4,
                        'timestamp': int(time.time() * 1000),  # 毫秒时间戳
                        'points_lidar': hatch_corners_refined.tolist(),
                        'points_world': points_world.tolist(),
                        'current_machine_position': 0,
                        'capture_point': {'x': 0, 'y': 0, 'z': 0},
                        'capture_point2': {'x': 0, 'y': 0, 'z': 0},
                        'capture_point_effective': 0,
                        'current_line_layer': 0,
                        'capture_point_layer': 0,
                        'capture_point2_layer': 0,
                        'capture_point_layer_min_height': 0.0,
                        'capture_point2_layer_min_height': 0.0,
                        'current_hatch': int(current_hatch),
                        'current_unLoadShip':3
                    }
               
                    #发送给我连接的java服务器
                    json_message = json.dumps(to_java_data, ensure_ascii=False)

                    await ws_manager.send_message(json_message)
                    print(f"发送给java后台的数据:{to_java_data}")

                    await broadcast_server.broadcast_coal_pile_data(coal_pile_data)

                    time.sleep(5)
                    depth_result = get_bottom_depth_result(process_id="170", current_hatch=current_hatch)
                    if depth_result:
                        bottom_depth = depth_result['bottom_depth']
                        is_depth_valid = depth_result['is_valid']
                        
                        if is_depth_valid:
                        
                            hatch_depth_data={
                            'type':5,
                            'current_hatch': int(current_hatch),
                            'hatch_depth': float(depth_result['hatch_depth']),
                                    }
                    
                            await ws_manager.send_message(json.dumps(hatch_depth_data, ensure_ascii=False))
                            print(f"发送给java后台的数据:{hatch_depth_data}")

                        else:
                            logger.info("获取到的舱底深度数据无效,不是舱底平面")

                    else:
                        logger.info("暂未获取到舱底深度数据")


                else:
                    # 检查是否为WebSocket相关异常
                    if isinstance(e, (websockets.exceptions.WebSocketException, 
                                    websockets.exceptions.ConnectionClosed,
                                    ConnectionError, OSError)):
                        logger.error(f"与java后台的连接已断开，错误信息: {e}", exc_info=True)
                        ws_manager.is_connected = False
                    else:    
                        logger.error(f"单次处理循环出错: {e}", exc_info=True)
                        # 发给java后台的数据，type=3 表示计算失败
                        to_java_data={
                            'type':3,
                            'timestamp': 0,
                            'points_lidar': [],
                            'points_world': [],
                            'current_machine_position': 0,
                            'capture_point': {'x': 0, 'y': 0, 'z': 0},
                            'capture_point2': {'x': 0, 'y': 0, 'z': 0},
                            'capture_point_effective': 0,
                            'current_line_layer': 0,
                            'capture_point_layer': 0,
                            'capture_point2_layer': 0,
                            'capture_point_layer_min_height': 0.0,
                            'capture_point2_layer_min_height': 0.0,
                            
                            'current_hatch': int(current_hatch),
                            'current_unLoadShip':3
                        }
                
                        # 发送给我连接的java服务器
                        json_message = json.dumps(to_java_data, ensure_ascii=False)
                        await ws_manager.send_message(json_message)
                        print(f"发送给java后台的数据:{to_java_data}")
 

                await asyncio.sleep(0.5)  # 等待0.5秒后继续下一次循环
                continue  # 继续下一次while循环
                
            #try块执行成功后，执行else块
            else:
                coal_pile_data["capture_point"] = capture_point
                coal_pile_data["capture_point_layer"]=capture_point_layer
                coal_pile_data["current_line_layer"]=current_line_layer
                coal_pile_data["capture_point_layer_min_height"]=float(capture_point_layer_min_height)
                coal_pile_data["square_ranges"]=square_ranges
                coal_pile_data["lines_dict_tojava"]=lines_dict_tojava
                
                
                 #发给java后台的数据,抓取点坐标，current_line_layer,capture_point_layer,min_height
                to_java_data={
                    'type':2,
                    'timestamp': int(time.time() * 1000),  # 毫秒时间戳
                    'points_lidar': hatch_corners_refined.tolist(),
                    'points_world': points_world.tolist(),
                    'current_machine_position': current_machine_position if hatch_recognize_success else last_machine_position,
                    'capture_point': capture_point,
                    'capture_point2':capture_point2,
                    'capture_point_effective':capture_point_effective,
                    'current_line_layer': int(current_line_layer),
                    'capture_point_layer': int(capture_point_layer),
                    'capture_point2_layer':int(capture_point2_layer),
                    'capture_point_layer_min_height': float(capture_point_layer_min_height),
                    'capture_point2_layer_min_height':float(capture_point2_layer_min_height),
                    'current_hatch': int(current_hatch),
                    'current_unLoadShip':3,
                    'mode_flag':mode_flag,
                    'blocks_heights':blocks_heights_list,
                    'current_coal_layer':int(current_layer),
                    'nine_square_block_heights':nine_square_block_heights,
                    'lines_dict_tojava':lines_dict_tojava,
                }
               
                #发送给我连接的java服务器
                json_message = json.dumps(to_java_data, ensure_ascii=False)

                await ws_manager.send_message(json_message)
                print(f"发送给java后台的数据:{to_java_data}")
                await broadcast_server.broadcast_coal_pile_data(coal_pile_data)
                print(f"煤堆数据已广播到所有连接的客户端")





              





              


            


            



            

            
            


                




            
            

            
    except KeyboardInterrupt:
        # print("\n程序被用户中断")
        logger.info("程序被用户中断")
    except Exception as e:
        # print(f"程序运行出错: {e}")
        logger.error(f"程序运行出错: {e}",exc_info= True)
        
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
        logger.error(f"程序运行出错: {e}",exc_info= True)


