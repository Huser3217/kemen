# 导入所需的库
from shapely.geometry import point
import laspy 
import numpy as np
import open3d as o3d 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from scipy.spatial.transform import Rotation as R
import time
import logging
import os
import sys
import json 

# 配置日志系统
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 获取进程标识符，如果没有传入则使用默认值
process_id = sys.argv[4] if len(sys.argv) > 4 else "default"
log_file_path = os.path.join(log_dir, f"measure_depth_{process_id}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    filename=log_file_path,
    filemode="a",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

def preprocess_point_cloud(pcd, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0):
    """
    点云预处理：下采样和去噪
    """
    logger.info(f"原始点云包含 {len(pcd.points)} 个点")
    
    # 下采样
    # pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    # logger.info(f"下采样后点云包含 {len(pcd_downsampled.points)} 个点")
    pcd_downsampled=pcd
    # 统计滤波去噪
    pcd_filtered, _ = pcd_downsampled.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    logger.info(f"去噪后点云包含 {len(pcd_filtered.points)} 个点")
    
    return pcd_filtered


def detect_bottom_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """
    使用RANSAC算法检测舱底平面
    返回: plane_model, inlier_points, outlier_points (numpy数组)
    """
    logger.info("正在检测舱底平面...")
    
    # 使用RANSAC检测平面
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    [a, b, c, d] = plane_model
    logger.info(f"平面方程: {a:.3f}x + {b:.3f}y + {c:.3}z + {d:.3f} = 0")
    logger.info(f"检测到 {len(inliers)} 个平面内点")
    
    # 提取平面点云和非平面点云
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    # 为可视化着色
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 红色表示平面
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色表示其他点
    
    # 转换为numpy数组
    inlier_points = np.asarray(inlier_cloud.points)
    outlier_points = np.asarray(outlier_cloud.points)
    
    return plane_model, inlier_points, outlier_points, inlier_cloud, outlier_cloud


def create_plane_mesh(plane_model, plane_points, extend_factor=1.2):
    """
    创建平面的3D网格用于可视化
    """
    points = np.asarray(plane_points.points)
    
    # 获取平面参数
    a, b, c, d = plane_model
    
    # 计算平面点的边界
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # 扩展边界
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * (extend_factor - 1) / 2
    x_max += x_range * (extend_factor - 1) / 2
    y_min -= y_range * (extend_factor - 1) / 2
    y_max += y_range * (extend_factor - 1) / 2
    
    # 创建网格点
    resolution = 20  # 网格分辨率
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # 根据平面方程计算Z坐标: ax + by + cz + d = 0 => z = -(ax + by + d)/c
    if abs(c) > 1e-6:  # 确保c不为0
        Z = -(a * X + b * Y + d) / c
    else:
        # 如果c接近0，平面垂直，使用平均Z值
        Z = np.full_like(X, np.mean(points[:, 2]))
    
    # 创建顶点
    vertices = []
    for i in range(resolution):
        for j in range(resolution):
            vertices.append([X[i, j], Y[i, j], Z[i, j]])
    
    # 创建三角形面
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # 每个网格单元创建两个三角形
            v1 = i * resolution + j
            v2 = i * resolution + (j + 1)
            v3 = (i + 1) * resolution + j
            v4 = (i + 1) * resolution + (j + 1)
            
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])
    
    # 创建Open3D网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 设置半透明的蓝色
    mesh.paint_uniform_color([0.3, 0.6, 1.0])  # 蓝色平面
    
    # 计算法向量
    mesh.compute_vertex_normals()
    
    return mesh


def create_plane_normal_vector(plane_model, plane_points, length=2.0):
    """
    创建平面法向量的可视化
    """
    points = np.asarray(plane_points.points)
    centroid = np.mean(points, axis=0)
    
    # 获取平面法向量
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # 归一化
    
    # 创建法向量线段
    start_point = centroid
    end_point = centroid + normal * length
    
    # 创建LineSet对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # 黄色法向量
    
    return line_set




def calculate_angle_with_z_axis(plane_model, threshold=15):
    """
    计算平面法向量与z轴的夹角，如果大于阈值则返回 False，否则 True
    plane_model: (a, b, c, d) 平面方程 ax + by + cz + d = 0
    threshold: 角度阈值（默认 15°）
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal)  # 归一化

    z_axis = np.array([0.0, 0.0, 1.0])  # z轴方向向量
    dot_val = np.dot(normal, z_axis)

    # 防止浮点误差超出 [-1, 1]
    dot_val = np.clip(dot_val, -1.0, 1.0)

    angle_deg = np.degrees(np.arccos(abs(dot_val)))  # 使用绝对值，因为法向量可能指向相反方向
    logger.info(f"平面与z轴的夹角: {angle_deg:.2f} 度")
    return angle_deg <= threshold

def validate_bottom_plane_by_height(points, height_threshold=5.0):
    """
    验证平面是否为舱底平面，通过检查点云在平面上的高度分布
    plane_model: 平面方程参数 [a, b, c, d]
    points: 点云数据 (N, 3)
    height_threshold: 高度阈值（默认 5.0 米）
    """
    if hasattr(points, 'points'):
        points_array = np.asarray(points.points)
    else:
        points_array = points
    z = points_array[:, 2]  # z坐标是高度
    max_height = np.max(z)
    min_height = np.min(z)
    height_diff = max_height - min_height
    logger.info(f"最大高度: {max_height:.2f} 米")
    logger.info(f"最小高度: {min_height:.2f} 米")
    logger.info(f"高度差: {height_diff:.2f} 米")
    return height_diff <= height_threshold


def load_coal_pile_data(data_file_path):
    """
    从文件加载煤堆点云数据
    """
    try:
        if os.path.exists(data_file_path):
            coal_pile_points = np.load(data_file_path)
            logger.info(f"成功加载煤堆点云数据，包含 {len(coal_pile_points)} 个点")
            return coal_pile_points
        else:
            logger.error(f"数据文件不存在: {data_file_path}")
            return None
    except Exception as e:
        logger.exception(f"加载煤堆点云数据失败: {e}")
        return None

def create_point_cloud_from_array(points_array):
    """
    从numpy数组创建Open3D点云对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    # 使用高度着色
    z_normalized = (points_array[:, 2] - points_array[:, 2].min()) / (points_array[:, 2].max() - points_array[:, 2].min())
    colors = plt.cm.viridis(z_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


if __name__ == "__main__":
    try:
        logger.info("开始船舱深度测量任务")
        
        # 检查是否传入了数据文件路径和舱口号
        if len(sys.argv) > 1:
            data_file_path = sys.argv[1]
            current_hatch = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 获取舱口号，默认为0
            hatch_height = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
            process_id = sys.argv[4] if len(sys.argv) > 4 else "default"  # 获取进程标识符
            
            logger.info(f"使用传入的煤堆点云数据: {data_file_path}")
            logger.info(f"当前舱口号: {current_hatch}")
            logger.info(f"当前舱高: {hatch_height:.2f} 米")
            logger.info(f"进程标识符: {process_id}")
            
            # 加载煤堆点云数据
            coal_pile_points = load_coal_pile_data(data_file_path)
            
            if coal_pile_points is not None:
                # 创建点云对象
                pcd = create_point_cloud_from_array(coal_pile_points)
                logger.info(f"创建点云对象成功，包含 {len(pcd.points)} 个点")
                
                # 清理临时文件
                try:
                    os.remove(data_file_path)
                    logger.info(f"已清理临时数据文件: {data_file_path}")
                except:
                    logger.warning(f"清理临时文件失败: {data_file_path}")
            else:
                logger.error("无法加载煤堆点云数据，退出程序")
                sys.exit(1)
        else:
            logger.error("未传入煤堆点云数据文件路径，退出程序")
            sys.exit(1)

        # 步骤1: 点云预处理
        pcd_processed = preprocess_point_cloud(pcd, voxel_size=0.1)
        
        # 步骤2: 检测舱底平面
        plane_model, plane_points, other_points, plane_cloud, other_cloud = detect_bottom_plane(
            pcd_processed, distance_threshold=0.05, num_iterations=5000)

        # 验证平面是否为舱底平面
        is_valid_bottom_plane = calculate_angle_with_z_axis(plane_model) & validate_bottom_plane_by_height(plane_points)
        
        if not is_valid_bottom_plane:
            logger.warning("检测到的平面可能不是舱底平面，建议调整参数重新检测！")
            # 即使验证失败，也保存结果供参考
            bottom_depth = np.mean(plane_points[:, 2])
            logger.info(f"舱底深度（未验证）: {bottom_depth:.2f} 米")
        else:
            logger.info("舱底平面验证通过")
            # 使用平面上所有的点的z坐标的平均值作为舱底深度
            bottom_depth = np.mean(plane_points[:, 2])
            logger.info(f"舱底深度: {bottom_depth:.2f} 米")

        hatch_depth = abs(hatch_height - bottom_depth)
        logger.info(f"型深: {hatch_depth:.2f} 米")

        # 保存结果到文件，供hatch_coal.py读取
        result_data = {
            'bottom_depth': float(bottom_depth),
            'is_valid': bool(is_valid_bottom_plane),
            'plane_model': plane_model.tolist(),
            'timestamp': time.time(),
            'point_count': len(plane_points),
            'current_hatch': current_hatch,  # 添加舱口号字段
            'hatch_depth': float(hatch_depth)  # 添加舱深字段
        }
        
        result_file_path = os.path.join(os.path.dirname(__file__), f"depth_measurement_result_{process_id}_{current_hatch}.json")
        try:
            with open(result_file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            logger.info(f"测量结果已保存到: {result_file_path}")
        except Exception as e:
            logger.exception(f"保存测量结果失败: {e}", exc_info=True)
        
        # # 创建平面网格用于可视化
        # plane_mesh = create_plane_mesh(plane_model, plane_cloud)
        # plane_normal = create_plane_normal_vector(plane_model, plane_cloud)
        
        # # 3D可视化：点云和平面
        # vis_geometries = [plane_cloud, other_cloud, plane_mesh, plane_normal]
        # o3d.visualization.draw_geometries(vis_geometries)
            logger.info("船舱深度测量任务完成")
        
    except Exception as e:
        logger.exception(f"处理过程中出现错误: {str(e)}", exc_info=True)
