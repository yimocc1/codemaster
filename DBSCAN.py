import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2


# yimocc1

class DensityCluster:
    def __init__(self, eps=0.5, min_samples=5):
        """
        定义核心参数
        """
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, data):
        """
        定义聚类函数
        Input:data
        Output:labels
        """
        self.cluster_labels = {}  # 初始化标签
        unique_timestamps = np.unique(data[:, 0])
        for timestamp in unique_timestamps:  # 每个time_step进行一次聚类
            timestamp_data = data[data[:, 0] == timestamp]  # 提取当前时刻行人样本
            coordinates = timestamp_data[:, 2:4]  # 提取行人坐标
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)  # 调用DBSCAN聚类模型
            labels = dbscan.fit_predict(coordinates)  # 预测当前时刻聚类结果
            self.cluster_labels[timestamp] = labels
        return self.cluster_labels  # 返回标签

    def generate_video(self, data, output_filename='output.mp4', fps=20):
        """
        以一定帧率播放聚类后的行人轨迹
        :param data:
        :param output_filename:
        :param fps:
        :return: None
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (640, 480)  # 帧大小
        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

        # 将坐标缩放至标准帧大小的相应位置
        min_x, min_y = np.min(data[:, 2:4], axis=0)
        max_x, max_y = np.max(data[:, 2:4], axis=0)
        data[:, 2] = (data[:, 2] - min_x) / (max_x - min_x) * frame_size[0]
        data[:, 3] = (data[:, 3] - min_y) / (max_y - min_y) * frame_size[1]

        color_map = {}
        color_counter = 0

        for timestamp in sorted(self.cluster_labels.keys()):
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            mask = data[:, 0] == timestamp
            labels = self.cluster_labels[timestamp]
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    col = (0, 0, 0)
                else:
                    if label not in color_map:
                        color = plt.cm.Spectral(color_counter / len(unique_labels))[:3]
                        color = tuple((np.array(color) * 255).astype(int))
                        color_map[label] = color
                        color_counter += 1
                    col = color_map[label]
                    col = tuple([int(x) for x in col])  # 设置为整数
                class_member_mask = (labels == label)
                xy = data[mask][class_member_mask]
                for point in xy:
                    cv2.circle(frame, (int(point[2]), int(point[3])), 5, col, -1)

            out.write(frame)

        out.release()
        print(f"Video saved as {output_filename}")


def read_data_from_file(file_path):
    """
    读取数据
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                time_step, ID, x, y = map(float, parts)
                data.append([time_step, ID, x, y])
    return np.array(data)


# Example usage
if __name__ == "__main__":
    file_path = 'students003.txt'  # 样本路径
    data = read_data_from_file(file_path)

    density_cluster = DensityCluster(eps=0.8, min_samples=1)  # 初始化类
    clusters = density_cluster.fit(data)  # 聚类
    print("Cluster labels by timestamp:", clusters)

    density_cluster.generate_video(data, output_filename='clusters.mp4', fps=10)  # 播放结果
