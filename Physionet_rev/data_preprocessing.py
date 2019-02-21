import pickle
import csv
import numpy as np
import copy

def get_fast_slow_features_per_ID(data, key, num_feats = 37):
    """
    Calculates the normalized sum of the delta_t of each feature PER ID and saves it to a csv file
    """
    
    with open("temp/33feat_output_scores.csv", "a", newline='') as fp:
        for ID in data[key]:
           # print('Running ID: '+ str(ID))
            data_matrix = np.matrix(data['data'][ID][0])
            missing_matrix = np.matrix(data['data'][ID][1])
            t = np.array(data['data'][ID][2])
            delta_matrix= np.full_like(missing_matrix, -1)
            #iterate through each feature
            for feat_ind in range(num_feats):
                current_feature = missing_matrix[:,feat_ind]
                delta_t = 0
                #iterate through current feature. if the missing_matrix says the value is missing, then add 1 to delta_t. if the value is not missing, reset delta_t to 0
                for i, ind in enumerate(current_feature):
                        flag = int(ind)
                        if(flag==0):
                            delta_t = 0 
                        delta_matrix[i,feat_ind] = delta_t
                        delta_t+=1

             #get sum of deltas
            delta_sums = delta_matrix.sum(axis = 0)
            #normalize this because in the future there will be variable length t's, so having a scale 0 to 1 is ideal
            delta_sums = delta_sums / np.sum(np.arange(len(t)))
            #delta_sums[delta_sums > 0.5] = 1
            #delta_sums[delta_sums <= 0.5] = 0
            y = delta_sums.tolist()[0]
            y.append(ID)
            print(y)
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(y)
                #f.write( str(ID) + ": " + str(delta_sums) + "\n" )



def get_fast_slow_features_entire_dataset(data, key, num_feats = 37):
    """
    Calculates the normalized sum of the delta_t of each feature to classify fast/slow features. Closer to 0 is a fast feature. Closer to 1 is a slow feature.
    """
    data_matrix = None
    for ID, data_tuple in data['data'].items():
        print('Running ID: '+ str(ID))
        if data_matrix is None:
            data_matrix = np.matrix(data_tuple[0])
            missing_matrix = np.matrix(data_tuple[1])
        else:
            current_data = np.matrix(data_tuple[0])
            current_missing = np.matrix(data_tuple[1])
            data_matrix = np.append(data_matrix,current_data,axis = 0)
            missing_matrix = np.append(missing_matrix,current_missing, axis = 0)
            
    print(data_matrix.shape)
    print(missing_matrix.shape)
            
        #t = np.array(data['data'][ID][2])
        
    
    delta_matrix= np.full_like(missing_matrix, -1)
    #iterate through each feature
    for feat_ind in range(num_feats):
        current_feature = missing_matrix[:,feat_ind]
        delta_t = 0
        #iterate through current feature. if the missing_matrix says the value is missing, then add 1 to delta_t. if the value is not missing, reset delta_t to 0
        for i, ind in enumerate(current_feature):
                flag = int(ind)
                if(flag==0):
                    delta_t = 0 
                delta_matrix[i,feat_ind] = delta_t
                delta_t+=1

     #get sum of deltas
    delta_sums = delta_matrix.sum(axis = 0)
    #normalize it
    delta_sums = delta_sums / np.sum(np.arange(missing_matrix.shape[0]))
    print(delta_sums)

def _split_data_fast_slow(data,num_features, fast_indexes, slow_indexes):
    """
    Returns fast/slow data the same structure as data
    """
    fast_data = copy.deepcopy(data)
    slow_data = copy.deepcopy(data)

    #NOTE: PYTHON2 is 
    #for ID, current_data in data.iteritems():
    #NOTE PYTHON3 is
    #for ID, current_data in data.items():
    list_data = data[0] #get the data as a list
    numpy_data = np.matrix(list_data) #convert it to numpy matrix

    fast_feats = numpy_data[:,fast_indexes]
    slow_feats = numpy_data[:,slow_indexes]

    list_missing = data[1]#get the missing matrix
    numpy_missing = np.matrix(list_missing)
    fast_missing = numpy_missing[:,fast_indexes]
    slow_missing = numpy_missing[:,slow_indexes]
    
    #print(fast_feats.shape)
    #print(slow_feats.shape)
    #some sanity checks
    assert fast_feats.shape[1] == len(fast_indexes)
    assert slow_feats.shape[1] == len(slow_indexes)

    #A tuple is immutable, so you need to create a new one
    #index 2 (timestamps) and index 3 (label) are unchanged
    fast_data = (fast_feats.tolist(), fast_missing.tolist(), fast_data[2], fast_data[3])
    slow_data = (slow_feats.tolist(), slow_missing.tolist(), slow_data[2], slow_data[3])

    return fast_data, slow_data

def _remove_missing_rows(data):
    """
    Removes rows that are all missing (missing flag is 1 for each feature)
    """
    deleted_ids = []
    #for ID, current_data in data.items():
    numpy_data = np.matrix(data[0])
    numpy_missing = np.matrix(data[1])
    timestamps = np.array(data[2])
    num_timestamps = numpy_data.shape[0]
    rows_to_remove = []
    for row in range(num_timestamps):
        current_row = numpy_missing[row,:]
        equal = np.array_equal(current_row, np.ones((1,numpy_data.shape[1])))
        if equal:
            rows_to_remove.append(row)

    numpy_data = np.delete(numpy_data, (rows_to_remove), axis=0)
    numpy_missing = np.delete(numpy_missing, (rows_to_remove), axis=0)
    timestamps = np.delete(timestamps,rows_to_remove)

    assert numpy_data.shape[0] == numpy_missing.shape[0]
    assert len(timestamps) == numpy_data.shape[0]

    
    return (numpy_data, numpy_missing, timestamps, data[3])


def create_split_dataset(data):
    fast_ids_with_zero = []
    slow_ids_with_zero = []
    for ID, current_data in data['data'].items():
        fast_zero_flag = 0
        slow_zero_flag = 0
        print('Running ID: ' + str(ID))
        fast_data, slow_data = _split_data_fast_slow(data['data'][ID], 37, [4,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,29,30,33,34,35,36], [0,1,2,3,5,6,16,27,28,31,32])
        fast_data = _remove_missing_rows(fast_data)
        slow_data = _remove_missing_rows(slow_data)
        if fast_data[0].shape[0] == 0:
            fast_ids_with_zero.append(ID)
            fast_zero_flag = 1
        elif slow_data[0].shape[0] == 0:
            slow_ids_with_zero.append(ID)
            slow_zero_flag = 1

        data['data'][ID] = (fast_data[0].tolist(), fast_data[1].tolist(), fast_data[2], fast_zero_flag ,slow_data[0].tolist(),slow_data[1].tolist(), slow_data[2],slow_zero_flag,fast_data[3])

    #for ID in fast_id_to_remove:
        #del data[ID]
    print("Fast Zero Number: " + str(len(fast_ids_with_zero)) + " Slow Zero Number: " + str(len(slow_ids_with_zero)) + " Total: " + str(len(fast_ids_with_zero) + len(slow_ids_with_zero)) )
    return data 


#data = pickle.load(open('data/final_Physionet_avg_new.pkl','rb'))
#new_data = create_split_dataset(data)
#pickle.dump(new_data, open('data/final_Physionet_avg_new_split.pkl','wb'), protocol=2)
data =  pickle.load(open('data/final_Physionet_avg_33feats_new.pkl','rb')) 
get_fast_slow_features_per_ID(data, 'train_ids', num_feats = 33)
 
