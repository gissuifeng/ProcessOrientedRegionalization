from fastdtw import fastdtw
import numpy as np
from sklearn.metrics import silhouette_score
import time



#  Read the distance matrix between STS
def get_dis_matrix_by_matrixFile():
    dis_matrix = np.zeros((zone_num,zone_num))
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/PM2.5_data/PM2.5_Euclidean_distance_between_zones.csv") as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            zone1,zone2,flow = line.split(",")
            zone1,zone2,flow = int(zone1),int(zone2),int(flow)
            dis_matrix[zone2,zone1] = dis_matrix[zone1,zone2] = flow
    return dis_matrix

# Read the similarity matrix of values between STS (similarity)
def get_sim_matrix_by_matrixFile():
    sim_matrix = np.zeros((zone_num,zone_num))
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/PM2.5_data/PM2.5_value_similarity_between_zones.csv", 'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            zone1,zone2,flow = line.split(",")
            zone1,zone2,flow = int(zone1),int(zone2),float(flow)
            sim_matrix[zone2,zone1] = sim_matrix[zone1,zone2] = flow
    return sim_matrix


# Obtain the similarity between two spatiotemporal sequences
def get_dis_two_STS(lst1, lst2):
    # 使用fastdtw计算两个时间序列之间的距离
    distance, path = fastdtw(lst1, lst2)
    return distance


# Read community information, #community numbers start from 0, unit numbers start from 0 after modification
def read_community_data(community_num):
    community_file_path = r"../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/PM2.5_data/Value_Similarity_Result_Data/result_"+str(community_num)+".csv"
    community_array = np.full(fill_value=-1,shape=(1113),dtype=int)
    with open(community_file_path, 'r') as fr:
        for cid,line in enumerate(fr):
            line = line.strip()
            attrs = [int(item) for item in line.split(",")]
            for unite in attrs:
                community_array[unite-1] = cid
    return community_array








#

# Calculate silhouette coefficient using distance matrix, considering both distance and simlarity
def silhouette_score_by_matrix(data, labels):
    n = len(data)
    silhouette_vals = []
    for i in range(n):
        # Calculate the average distance from sample i to other samples in the same cluster, smaller a is better,
        # maximum similarity/distance: 13775.0, maximum length: 1980942.2819010408
        a = sum([dis_matrix[i,j]/13776 + sim_matrix[i,j]/1980943 for j in range(n) if labels[j] == labels[i]]) / (sum(labels == labels[i]) - 1)

        # Calculate the average distance from sample i to the nearest cluster and find the minimum value, larger b is better
        b = min(
            [sum([dis_matrix[i,j]/13776 + sim_matrix[i,j]/1980943 for j in range(n) if labels[j] != labels[i]]) / sum(labels != labels[i]) for
             j in range(n) if labels[i] != labels[j]])

        # Calculate the silhouette coefficient for sample i
        silhouette_val = round((b - a) / max(a, b) if max(a, b) != 0 else 0, 3)
        silhouette_vals.append(silhouette_val)
        # print(i,silhouette_val)
    # Return the average silhouette coefficient of all samples
    return sum(silhouette_vals) / n




zone_num = 1113     # Number of zones
# Read the distance matrix between STS
dis_matrix = get_dis_matrix_by_matrixFile()
# Read the similarity matrix between STS
sim_matrix = get_sim_matrix_by_matrixFile()


for community_num in range(12,18):
    # Read community information
    end_time1 = time.time()
    labels_array = read_community_data(community_num)
    end_time2 = time.time()
    print("Community data read completed in", round(end_time2 - end_time1, 3), "seconds")

    score = silhouette_score_by_matrix(range(zone_num), labels_array)
    print("Number of communities:", community_num, ", Silhouette coefficient:", score)

    end_time3 = time.time()
    print("Program executed in", round(end_time3 - end_time2, 3), "seconds")








