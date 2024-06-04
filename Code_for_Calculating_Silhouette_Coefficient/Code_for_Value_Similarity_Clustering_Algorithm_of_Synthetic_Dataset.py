import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Sample data
# data_dict = {
#     'zone1': np.random.rand(10),
#     'zone2': np.random.rand(10),
#     'zone3': np.random.rand(10),
#     'zone4': np.random.rand(10),
#     'zone5': np.random.rand(10)
# }
# Read spatio-temporal data
def get_data_dict():
    data_dict = {}
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/Synthetic_dataset/Value_similarity_data2.csv", 'r') as fr:
        for line in fr:
            line.strip()
            value_list = [float(item) for item in line.split(",")]
            data_dict[str(int(value_list[0]))] = np.array(value_list[2:])
    return data_dict



#Read neighbor information
def get_neigobor_byFile():
    neighbor_dict = {}
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/Synthetic_dataset/Neighbor_information.csv") as fr:
        for line in fr:
            line = line.strip()
            zone_list = [zone for zone in line.split(",")]
            neighbor_dict[zone_list[0]] = zone_list[1:]
    return neighbor_dict
    # 示例邻居信息
    #     neighbor_dict = {
    #         'zone1': ['zone2'],
    #         'zone2': ['zone1', 'zone3'],
    #         'zone3': ['zone2', 'zone4'],
    #         'zone4': ['zone3', 'zone5'],
    #         'zone5': ['zone4']
    #     }

# Calculate the value similarity between the feature lists of two districts/areas, using the DTW (Dynamic Time Warping) method.
def calculate_distance(zone1_features, zone2_features):
    distance, _ = fastdtw(zone1_features, zone2_features)
    # print(distance)
    return distance


# Calculate the average feature list for a group of districts/areas.
def calculate_mean_features(zone_list, data_dict):
    # print(zone_list)
    mean_features = np.mean([data_dict[zone] for zone in zone_list], axis=0)
    # print("average:", mean_features)
    return mean_features
# Agglomerative hierarchical clustering algorithm
def hierarchical_clustering(data_dict, neighbor_dict, k):
    zone_list = list(data_dict.keys())  # Initially, each district/area is a separate district/area group on its own.
    while len(zone_list) > k:  # Until the specified number of partitions is reached.
        min_distance = float('inf')
        merge_zone_pair = None
        for i in range(len(zone_list)):
            for j in range(i + 1, len(zone_list)):
                if zone_list[j] in neighbor_dict[zone_list[i]]:
                    distance = calculate_distance(data_dict[zone_list[i]], data_dict[zone_list[j]])
                    # print("##",distance)
                    if distance < min_distance:
                        min_distance = distance
                        merge_zone_pair = (zone_list[i], zone_list[j])
        if len(zone_list)<4:
            print(zone_list)


        # Merge the two closest areas into one district/area group
        merged_zone = [merge_zone_pair[0], merge_zone_pair[1]]
        merged_zone_str = ','.join(merged_zone)
        merged_zone_str = merged_zone_str.strip()
        # mean_features = calculate_mean_features(merged_zone, data_dict)
        mean_features = calculate_mean_features(merged_zone_str.split(","), data_dict)
        data_dict[merged_zone_str] = mean_features

        # update neighbor information
        merged_neighbors = set(neighbor_dict[merge_zone_pair[0]]) | set(neighbor_dict[merge_zone_pair[1]])
        neighbor_dict[merged_zone_str] = list(merged_neighbors)
        for neighbor in merged_neighbors:
            neighbor = neighbor.strip()
            # neighbor_dict[neighbor].remove(merge_zone_pair[0])
            # neighbor_dict[neighbor].remove(merge_zone_pair[1])
            neighbor_dict[neighbor].append(merged_zone_str)

        # Remove the merged district/area groups
        zone_list.remove(merge_zone_pair[0])
        zone_list.remove(merge_zone_pair[1])
        zone_list.append(merged_zone_str)
        print("Number of remaining nodes：",len(zone_list),"---currently merged nodes:",merged_zone_str)

    # 整理分区结果，将区域组内的区域列表提取出来
    clustered_zones = [zone.split(",") for zone in zone_list]

    return clustered_zones






# 读取空间时序
data_dict = get_data_dict()
# 示例数据
# data_dict = {
#     'zone1': np.random.rand(10),
#     'zone2': np.random.rand(10),
#     'zone3': np.random.rand(10),
#     'zone4': np.random.rand(10),
#     'zone5': np.random.rand(10)
# }

#Read neighbor information
neighbor_dict =  get_neigobor_byFile()
# Example neighbor information
# {
#     'zone1': ['zone2'],
#     'zone2': ['zone1', 'zone3'],
#     'zone3': ['zone2', 'zone4'],
#     'zone4': ['zone3', 'zone5'],
#     'zone5': ['zone4']
# }

# Specify the number of partitions
k = 4

# Perform agglomerative hierarchical clustering
clustered_zones = hierarchical_clustering(data_dict, neighbor_dict, k)
print("Partition results:", clustered_zones)
