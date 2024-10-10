import arcpy
import csv

def script_tool(featureclass, region_id, output_neighbor_metrix_file):
    """Script code goes below"""

    # 创建一个空字典来存储邻居关系
    neighbors = {}

    # 使用SearchCursor遍历所有要素
    with arcpy.da.SearchCursor(featureclass, ["SHAPE@", region_id]) as cursor:
        for row in cursor:
            current_shape = row[0]
            current_id = row[1]
            
            # 如果当前ID还没有在字典中，则添加它
            if current_id not in neighbors:
                neighbors[current_id] = set()
            
            # 再次遍历所有要素以找到邻居
            with arcpy.da.SearchCursor(featureclass, ["SHAPE@", region_id]) as cursor2:
                for row2 in cursor2:
                    neighbor_shape = row2[0]
                    neighbor_id = row2[1]
                    
                    # 如果不是同一个要素且有交集，则认为是邻居
                    if current_id != neighbor_id and current_shape.touches(neighbor_shape):
                        neighbors[current_id].add(neighbor_id)

    # 将结果写入CSV文件
    with open(output_neighbor_metrix_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for region_id, neighbor_ids in neighbors.items():
            row = [region_id] + list(neighbor_ids)
            writer.writerow(row)

    arcpy.AddMessage("Processing task has been successfully completed, and neighborhoods file has been exported!")

    return


if __name__ == "__main__":

    featureclass = arcpy.GetParameterAsText(0)
    region_id = arcpy.GetParameterAsText(1)
    output_neighbor_metrix_file = arcpy.GetParameterAsText(2)

    script_tool(featureclass, region_id, output_neighbor_metrix_file)