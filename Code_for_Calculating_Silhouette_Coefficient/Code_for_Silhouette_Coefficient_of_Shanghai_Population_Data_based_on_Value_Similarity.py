from fastdtw import fastdtw
import numpy as np
import time


# 读取社区信息
def read_community_data(community_num):
    community_file_path = r"../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/Shanghai_data_partition_results/result-"+str(community_num)+".csv"
    # Create a community list [[Community 1], [Community 2], ...]
    com_list = [[] for _ in range(community_num)]
    with open(community_file_path, 'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            com_id, zone_id = line.split(",")
            com_id, zone_id = int(com_id), int(zone_id)-1
            com_list[com_id-1].append(zone_id)
    return com_list

# Read spatial temporal data, with unit numbering starting from 0
def read_time_series_data(file_path):
    # Create a 1000 rows, 24 columns array to store population data of 1000 areas in Shanghai with 24 decimals
    data_list = [[] for _ in range(zone_num)]
    with open(file_path, 'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            region_id, hour, pop_sum = line.split(",")
            region_id, pop_sum = int(region_id), int(float(pop_sum))
            data_list[region_id-1].append(pop_sum)
    return  data_list



# Convert the community format [[Community 1], [Community 2], ...] to label format [5, 4, 4, 1, ...], where the first 5 represents an area belonging to Community 5,
def trans_community_data_to_labels(com_list):
    labels_array = np.zeros(zone_num )
    for index, lst in enumerate(com_list):  #index represents the community number, lst represents the list of units in the community with the index
        for item in lst:
            labels_array[item] = index + 1
    # print(labels_array)
    return labels_array


# Read the Euclidean distance matrix between STS.
def get_dis_matrix_by_matrixFile():
    dis_matrix = np.zeros((zone_num,zone_num))
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/File_containing_Euclidean_distances_between_areas_in_Shanghai.csv") as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            zone1,zone2,flow = line.split(",")
            zone1,zone2,flow = int(zone1),int(zone2),int(flow)
            dis_matrix[zone2,zone1] = dis_matrix[zone1,zone2] = flow
    return dis_matrix

# Read the similarity matrix of values between STS
def get_sim_matrix_by_matrixFile():
    sim_matrix = np.zeros((zone_num,zone_num))
    with open("../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/File_containing_value_similarity_between_areas_in_Shanghai.csv", 'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            zone1,zone2,flow = line.split(",")
            zone1,zone2,flow = int(zone1),int(zone2),float(flow)
            sim_matrix[zone2,zone1] = sim_matrix[zone1,zone2] = flow
    return sim_matrix


# Obtain the value similarity between two STSs
def get_dis_two_STS(lst1, lst2):
    # Calculate the value similarity between two time series using fastdtw
    distance, path = fastdtw(lst1, lst2)
    return distance





# Calculate silhouette coefficient using distance matrix, considering both distance and similarity
def silhouette_score_by_matrix(data, labels):
    n = len(data)
    silhouette_vals = []
    for i in range(n):
        # Calculate the average distance from sample i to other samples in the same cluster, smaller a is better, maximum similarity/distance: 2713981.0, maximum length: 33850.661989143766
        a = sum([dis_matrix[i,j]/2713982 + sim_matrix[i,j]/33851 for j in range(n) if labels[j] == labels[i]]) / (sum(labels == labels[i]) - 1)
        # Calculate the average distance from sample i to the nearest cluster and find the minimum value
        b = min(
            [sum([dis_matrix[i,j]/2713982 + sim_matrix[i,j]/33851 for j in range(n) if labels[j] != labels[i]]) / sum(labels != labels[i]) for
             j in range(n) if labels[i] != labels[j]])
        # Calculate the silhouette coefficient for sample i
        silhouette_val = round((b - a) / max(a, b) if max(a, b) != 0 else 0, 3)
        silhouette_vals.append(silhouette_val)
        # print(i,silhouette_val)
    # Return the average silhouette coefficient of all samples
    return sum(silhouette_vals) / n




zone_num = 1000  # Number of zones
attr_num = 24  # Number of values in a zone's time series record
# Path to spatial temporal data
file_path = r"../dataset/Data_for_calculating_the_silhouette_coefficient_of_the_result/24-hour_population_data_for_each_area_of_Shanghai.csv"  # _test
# Shape of spatial temporal data, zone_num represents the number of zones, attr_num represents the number of population time series in each zone
list_shape = (zone_num, attr_num)
# Read spatial temporal data.
data_list = read_time_series_data(file_path)
start_time = time.time()

# Read the Euclidean distance matrix between STS.
dis_matrix = get_dis_matrix_by_matrixFile()
# Read the similarity matrix of values between STS
sim_matrix = get_sim_matrix_by_matrixFile()

end_time1 = time.time()
print("dis_matrix read completed", round(end_time1 - start_time, 3))



for community_num in range(6,19):
    # Read community information
    com_list = read_community_data(community_num)
    end_time2 = time.time()
    print("Community data read completed in", round(end_time2 - end_time1, 3), "seconds.")

    labels_array = trans_community_data_to_labels(com_list)
    # score = silhouette_score(data_list, labels_array)
    score = silhouette_score_by_matrix(range(zone_num), labels_array)
    print("Number of communities:", community_num, ", Silhouette coefficient:", score)

    end_time3 = time.time()
    print("Program executed in", round(end_time3 - end_time2, 3), "seconds.")



