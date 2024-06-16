import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2


class DensityCluster:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize the density clustering algorithm with DBSCAN.

        Parameters:
        eps: float, default=0.5
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: int, default=5
            The number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, data):
        """
        Fit the DBSCAN algorithm on the provided data.

        Parameters:
        data: array-like, shape (n_samples, 4)。
            Input data, where each row is (timestamp, ID, x, y).

        Returns:
        clusters: dict
            A dictionary where the key is the timestamp and the value is the cluster labels for that timestamp.
        """
        self.cluster_labels = {}
        unique_timestamps = np.unique(data[:, 0])
        for timestamp in unique_timestamps:
            timestamp_data = data[data[:, 0] == timestamp]
            coordinates = timestamp_data[:, 2:4]
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(coordinates)
            self.cluster_labels[timestamp] = labels
        return self.cluster_labels

    def generate_video(self, data, output_filename='output.mp4', fps=20):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (640, 480)
        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

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
    Read data from a TXT file.

    Parameters:
    file_path: str
        Path to the TXT file containing the data.

    Returns:
    data: numpy array, shape (n_samples, 4)
        Array containing the data read from the file, with columns (time_step, ID, x, y).
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
    file_path = 'students003.txt'  # Replace with your actual file path
    data = read_data_from_file(file_path)

    density_cluster = DensityCluster(eps=0.8, min_samples=1)
    clusters = density_cluster.fit(data)
    print("Cluster labels by timestamp:", clusters)

    density_cluster.generate_video(data, output_filename='clusters.mp4',fps=10)
