import numpy as np
import pickle
import os

# Declare patient variables as global
tsteps_slow = np.empty([1,1])
tsteps_moderate = np.empty([1,1])
tsteps_fast = np.empty([1,1])
fast_data = np.empty([1,1])
moderate_data = np.empty([1,1])
slow_data = np.empty([1,1])

# Flags and Paths
filename = "mlhc_preprocessed_residual_fractions_1.pkl"
datapath = os.path.join("..","..","Data","final_Physionet_avg_new_split_threeway.pkl")
savepath = os.path.join("..", "..", "Data", filename)
fast_feature_indices = [15, 34, 31, 11]
moderate_feature_indices = [14, 8, 5, 13, 16, 27, 24, 20, 35, 30, 9, 18, 12, 23, 21, 22, 36, 37, 10, 26, 25, 19]
slow_feature_indices = [17, 28, 29, 6, 3, 2, 1, 4, 33, 7, 32]
subtract_data = False
frac = 1


def get_average_values(time_range, frequency):
    '''Get average of all features of a frequency braket, for the given timesteps'''
    global fast_data
    global moderate_data
    global tsteps_moderate
    global tsteps_fast

    if frequency == 'moderate':
        tsteps = tsteps_moderate
        data = moderate_data
    if frequency == 'fast':
        tsteps = tsteps_fast
        data = fast_data
        
    temp = []
    for j in time_range:
        if j in tsteps:

            # Get index of j in tsteps
            ind = int(np.where(tsteps == j)[0])

            # Get data from all of the block's features at that timestep
            temp.append(data[ind])

    # Average all feature values seen and save to append to slow features at index i
    temp = np.asarray(temp)         # (t, d_block)
#     print('vals used for avg', temp.shape)
    avg = np.average(temp, axis=0)  # (d_block,)
            
    return avg

def subtract_average(time_range, avg, frequency):
    global fast_data
    global moderate_data
    global tsteps_moderate
    global tsteps_fast

    if frequency == 'fast':
        tsteps = tsteps_fast
        data = fast_data
    if frequency == 'moderate':
        tsteps = tsteps_moderate
        data = moderate_data
    
    for j in time_range:
        if j in tsteps:

            # Get index of j in tsteps
            ind = int(np.where(tsteps == j)[0])

            # Get data from all of the block's features at that timestep
            data[ind] = data[ind] - avg

    return data

def prepare_data_block(slow_block_flag, all_tsteps):
    global fast_data
    global moderate_data
    global slow_data
    global tsteps_slow
    global tsteps_moderate
    global tsteps_fast

    if slow_block_flag:   # If frequency of current block is slow
        # print (fast_data.shape, moderate_data.shape, slow_data.shape, len(tsteps_slow))
        final_data = np.empty([len(tsteps_slow), fast_data.shape[1]+moderate_data.shape[1]+slow_data.shape[1]])
        tsteps = tsteps_slow
    else:
        final_data = np.empty([len(tsteps_moderate), fast_data.shape[1]+moderate_data.shape[1]])
        tsteps = tsteps_moderate
    
    last_seen = 0
    fill_index = 0
    for i in range(len(all_tsteps)):
        if all_tsteps[i] in tsteps:
        
            # Save all timesteps where features must be averaged
            time_range = all_tsteps[last_seen : i+1]
            last_seen = i+1
        
            # Find values in faster blocks from this time range, and average them
            if slow_block_flag:
                avg_moderate = frac * get_average_values(time_range, 'moderate')  # (d_moderate,)
                if np.isnan(avg_moderate).any():
                    avg_moderate = frac * np.average(moderate_data, axis=0)
            avg_fast = frac * get_average_values(time_range, 'fast')  # (d_fast,)
        
            # If there is no fast signal present, impute with average of signal
            if np.isnan(avg_fast).any():
                avg_fast = frac * np.average(fast_data, axis=0)
        
            # Remove averages
            if subtract_data:
                if slow_block_flag:
                    moderate_data = subtract_average(time_range, avg_moderate, 'moderate')
                fast_data = subtract_average(time_range, avg_fast, 'fast')

            # Concatenate averages from faster blocks with current block
            if slow_block_flag:
                temp = np.expand_dims(slow_data[int(np.where(tsteps_slow == all_tsteps[i])[0])], axis=1)        
                timestep_feature_vector = np.vstack((temp, np.expand_dims(avg_moderate, axis=1), np.expand_dims(avg_fast, axis=1)))  #(37, 1)
            else:
                temp = np.expand_dims(moderate_data[int(np.where(tsteps_moderate == all_tsteps[i])[0])], axis=1)        
                timestep_feature_vector = np.vstack((temp, np.expand_dims(avg_fast, axis=1)))  #(26, 1)
        
            # Add to final_slow_data
            final_data[fill_index] = np.squeeze(timestep_feature_vector)
            fill_index += 1
                
    return final_data

def get_final_data(patient_data):
    global fast_data
    global moderate_data
    global slow_data
    global tsteps_slow
    global tsteps_moderate
    global tsteps_fast

    # Save all data as numpy arrays of size (t_block, d_block)
    fast_data = np.array(patient_data[0])
    moderate_data = np.array(patient_data[4])
    slow_data = np.array(patient_data[8])
    print ("Fast, moderate and slow data: ", fast_data.shape, moderate_data.shape, slow_data.shape)

    # Save timesteps each feature has observations for
    tsteps_slow = np.array(patient_data[10])
    tsteps_moderate = np.array(patient_data[6])
    tsteps_fast = np.array(patient_data[2])

    # Merge all timesteps where data is observed
    all_tsteps = sorted(set(tsteps_slow.tolist() + tsteps_moderate.tolist() + tsteps_fast.tolist()))
    final_slow = prepare_data_block(True, all_tsteps)
    final_moderate = prepare_data_block(False, all_tsteps)
    final_fast = fast_data
    print("Final Data: ", final_slow.shape, final_moderate.shape, final_fast.shape)
    
    return final_slow, final_moderate, final_fast

########################################################################################################################

# Load Data
data_raw = pickle.load(open(datapath,'rb'))
print ("==x=="*20)
print ("Data Statistics")
print ("Train Data: "+str(len(data_raw['train_ids'])))
print ("Val Data: "+str(len(data_raw['val_ids'])))
print ("Test Data: "+str(len(data_raw['test_ids'])))
print ("==x=="*20)

# data['data'][id] = (fast_data, fast_missing, fast_timesteps, fast_zero_flag, moderate_data, moderate_missing, moderate_timesteps, moderate_zero_flag, slow_data, slow_missing, slow_timesteps, slow_zero_flag, label)

print("\nProcessing Training Data\n")
final_data_train = []
for i in range(len(data_raw['train_ids'])):
    idx = data_raw['train_ids'][i]
    patient_data = data_raw['data'][idx]
    label = patient_data[12]

    # If there is no data from a particular block, skip the patient
    if len(patient_data[10]) == 0 or len(patient_data[6]) == 0 or len(patient_data[2]) == 0:
        continue

    print("\nID: ", idx)
    final_slow, final_moderate, final_fast = get_final_data(patient_data)
    final_data_train.append((idx, final_slow, final_moderate, final_fast, label))

print("\n\n\nProcessing Validation Data\n")
final_data_val = []
for i in range(len(data_raw['val_ids'])):
    idx = data_raw['val_ids'][i]
    patient_data = data_raw['data'][idx]
    label = patient_data[12]

    # If there is no data from a particular block, skip the patient
    if len(patient_data[10]) == 0 or len(patient_data[6]) == 0 or len(patient_data[2]) == 0:
        continue

    print("\nID: ", idx)
    final_slow, final_moderate, final_fast = get_final_data(patient_data)
    final_data_val.append((idx, final_slow, final_moderate, final_fast, label))

print("\n\n\nProcessing Testing Data\n")
final_data_test = []
for i in range(len(data_raw['test_ids'])):
    idx = data_raw['test_ids'][i]
    patient_data = data_raw['data'][idx]
    label = patient_data[12]

    # If there is no data from a particular block, skip the patient
    if len(patient_data[10]) == 0 or len(patient_data[6]) == 0 or len(patient_data[2]) == 0:
        continue

    print("\nID: ", idx)
    final_slow, final_moderate, final_fast = get_final_data(patient_data)
    final_data_test.append((idx, final_slow, final_moderate, final_fast, label))


# After looping over all patients, pickle data
with open(savepath, 'wb') as f:
    pickle.dump([final_data_train, final_data_val, final_data_test] , f, protocol=2)

print("Data saved.")  # Now loading...")
# with open('processed_data.pkl', 'rb') as f:
#     loaded_obj = pickle.load(f)
# print("Loaded. Data is same check: ", (final_data == loaded_obj))

