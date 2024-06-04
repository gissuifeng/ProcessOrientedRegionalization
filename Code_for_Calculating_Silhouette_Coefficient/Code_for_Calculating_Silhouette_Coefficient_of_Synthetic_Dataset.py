from fastdtw import fastdtw
import numpy as np
import time
import math



# Read the similarity matrix between STS
def get_len_dict_by_dictFile():
    len_dict= {}
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/Synthetic_dataset/Similarity_file_of_values_between_zone_sequences.csv", 'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            zone1,zone2,len = line.split(",")
            len = float(len)
            len_dict[zone1+':'+zone2] = len_dict[zone2+':'+zone1] = len
    return len_dict

# Read the Euclidean distance matrix between the corresponding zones of STS
def get_dis_dict_by_dictFile():
    dis_dict = {}
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/Synthetic_dataset/file_for_distance_between_zones.csv") as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            zone1,zone2,dis = line.split(",")
            dis = int(dis)
            dis_dict[zone1+':'+zone2] = dis_dict[zone2+':'+zone1] = dis
    return dis_dict

# 读取社区信息，Synthetic_dataset默认4个分区
def read_community_data(cluster_num):
    community_file_path = r"../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/Synthetic_dataset/分区结果数据集/社区标签结果-"+str(cluster_num)+".csv"
    com_list = [-1]
    with open(community_file_path, 'r') as fr:
        fr.readline()  # 跳过标题行
        for line in fr:
            line = line.strip()
            zone_id, com_id = line.split(",")
            com_id = int(com_id)
            com_list.append(com_id)
    labels_array = np.array(com_list)
    return labels_array



# Calculate silhouette coefficient using distance matrix
def silhouette_score_by_matrix_with_neighbor(zone_num, labels):
    n = zone_num
    silhouette_vals = []

    for i in range(1,n+1):
        # Maximum distance: 706407.0, maximum similarity: 1148.98
        # Calculate the average distance from sample i to other samples in the same cluster, smaller a is better
        a = sum([dis_dict[str(i)+":"+str(j)]/706407 + len_dict[str(i)+":"+str(j)]/ 1149  for j in range(1,n+1) if labels[j] == labels[i] and i !=j ] ) / (sum(labels == labels[i]) - 1)#+ len_dict[str(i)+":"+str(j)]/ 1149
        # Calculate the average distance from sample i to the nearest cluster and find the minimum value
        b = min(
            [sum([dis_dict[str(i)+":"+str(j)]/706407 + len_dict[str(i)+":"+str(j)]/ 1149 for j in range(1,n+1) if labels[j] != labels[i] and i !=j ]) / sum(labels != labels[i]) for
             j in range(n) if labels[i] != labels[j]])

        # Calculate the silhouette coefficient for sample i
        silhouette_val = round((b - a) / max(a, b) if max(a, b) != 0 else 0, 3)
        if math.isnan(silhouette_val):
            silhouette_val = 0
        silhouette_vals.append(silhouette_val)
        # print(i,silhouette_val)
    # Return the average silhouette coefficient of all samples.
    return sum(silhouette_vals) / zone_num






zone_num = 79     #Number of zones


# Read the distance matrix between STS
dis_dict = get_dis_dict_by_dictFile()

# Read the similarity matrix  between STS
len_dict = get_len_dict_by_dictFile()

for cluster_num in range(2,10):
    # # Read community information, with Synthetic_dataset defaulting to 4 partitions
    labels_array = read_community_data(cluster_num)

    score = silhouette_score_by_matrix_with_neighbor(zone_num, labels_array)
    print("Number of communities:", cluster_num, ", Silhouette coefficient:", score)

    

#The synthetic dataset is set to have 4 clusters, and it can be observed that the silhouette coefficient is maximized when the result is 4 clusters.
