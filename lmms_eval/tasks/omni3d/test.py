import numpy as np


def convert_normalized_angles_to_rad(bbox_3d):
    """
    将归一化的角度值转换为弧度
    假设输入的最后三个值是相对于π的归一化值
    
    Args:
        bbox_3d: 包含9个元素的数组 [x, y, z, w, h, l, roll, pitch, yaw]
                最后三个值是归一化的角度值
    
    Returns:
        转换后的数组，最后三个值为弧度
    """
    # 复制数组避免修改原始数据
    result = list(bbox_3d)
    
    # 将最后三个值（归一化的角度）乘以180，然后转换为弧度
    for i in range(-3, 0):
        normalized_angle = bbox_3d[i]
        degrees = normalized_angle * 180
        radians = np.deg2rad(degrees)
        result[i] = radians
    
    return result

x = [-0.84, 0.29, 1.55, 0.97, 0.81, 1.53, 0.55, 0.36, 0.58]
print(convert_normalized_angles_to_rad(x))
