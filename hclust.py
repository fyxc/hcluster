# import module
import sys
import numpy as np
import heapq
import re
from itertools import groupby, combinations


# def func
# distance func
def distance(item1, item2):
    Dist = np.sum((np.array(item1) - np.array(item2)) ** 2)
    return Dist

# generate merged_p(point should be put in the same cluster)

def merge_p(merge_point):
    merge_p = []
    merge_index = [int(i) for i in re.findall(pattern = "\d+", string = str(merge_point))]
    for i in merge_index:
        merge_p.append(val[i])
    return merge_p

# new center after merge
def center(merged_list):
    ncenter = np.sum(merged_list, axis=0).astype("float64") / len(merged_list)
    return ncenter

# generate initial heap structure
def init_heap(data):
    h = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            tempdist = distance(data[i],data[j])
            heapq.heappush(h, [tempdist, ((i, j))])
    return h


def hcluster(h,data,k,dic):
    valid = set(range(len(data)))
    #for i in range(len(data)):
     #   valid.add((i,))
    length = len(data)
    # generate cluster
    while h:
        top_point = heapq.heappop(h)
        #merge_point = sum(top_point[1],()) # merge two tuples in to one, eg: ((a,b),(c,)) to (a,b,c)
        test_point = top_point[1]

        # if the testpoint is not in the valid dataset, ignore it
        if test_point[0] in valid and test_point[1] in valid:

            # find the center of new cluster
            merged_list = merge_p(test_point)
            newpoint = center(merged_list)
            dic[test_point] = newpoint
            valid.remove(test_point[0])
            valid.remove(test_point[1])
            length -= 1

            #calculate the dis bet new cluster and existed culster
            for p in valid:
                tempdist1 = distance(dic[p], newpoint)
                heapq.heappush(h,[tempdist1,(test_point, p)])
            valid.add(test_point)
            if length == k:
                return list(valid)


def output_eval(cluster, label_list):
    intersect = 0
    total_precision = 0
    pairs = [combinations(label_list[i][1], 2) for i in range(len(label_list))]
    pairs = [list(ele) for ele in pairs]
    Total_pairs = []
    for m in pairs:
        for n in m:
            Total_pairs.append(n)

    for i in range(len(cluster)):
        result = sorted([int(k) for k in re.findall(pattern="\d+", string=str(cluster[i]))])
        print ("Cluster{}:{}".format(i + 1, result))
        intersect += len(set(combinations(result, 2)).intersection(Total_pairs))
        total_precision += len(set(combinations(result, 2)))

    print("Precision = {}, Recall = {}".format(float(intersect) / total_precision,
                                                   float(intersect) / len(Total_pairs)))


# main func

if __name__ == "__main__":


    with open(sys.argv[1]) as f:
        data = [line.split(",") for line in f]
        str_val = [ele[0:4] for ele in data]
        val = [list(map(float, ele1)) for ele1 in str_val]
        label = [ele[-1].strip() for ele in data]
        f.close()
    # translate value list to dictionary
    dic = {}
    for i in range(len(val)):
        dic[i] = val[i]

    label_index = []
    for i in range(len(label)):
        label_index.append((label[i], i))

    k = sys.argv[2]
    heap = init_heap(val)
    cluster = hcluster(heap,val,int(k),dic)
    label_list = [(key, [i[1] for i in values]) for key, values in groupby(label_index, lambda i: i[0])]
    # print result
    output_eval(cluster,label_list)







