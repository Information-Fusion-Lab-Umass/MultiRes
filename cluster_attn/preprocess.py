import numpy as np
import pandas as pd

inputpath = './set-a/'
inputdict = {
    "ALP": 0,  # o
    "ALT": 1,  # o
    "AST": 2,  # o
    "Albumin": 3,  # o
    "BUN": 4,  # o
    "Bilirubin": 5,  # o
    "Cholesterol": 6,  # o
    "Creatinine": 7,  # o
    "DiasABP": 8,  # o
    "FiO2": 9,  # o
    "GCS": 10,  # o
    "Glucose": 11,  # o
    "HCO3": 12,  # o
    "HCT": 13,  # o
    "HR": 14,  # o
    "K": 15,  # o
    "Lactate": 16,  # o
    "MAP": 17,  # o
    "Mg": 18,  # o
    "Na": 19,  # o
    "PaCO2": 20,  # o
    "PaO2": 21,  # o
    "Platelets": 22,  # o
    "RespRate": 23,  # o
    "SaO2": 24,  # o
    "SysABP": 25,  # o
    "Temp": 26,  # o
    "Tropl": 27,  # o
    "TroponinI": 27,  # temp: regarded same as Tropl
    "TropT": 28,  # o
    "TroponinT": 28,  # temp: regarded same as TropT
    "Urine": 29,  # o
    "WBC": 30,  # o
    "Weight": 31,  # o
    "pH": 32,  # o
    "NIDiasABP": 33,  # unused variable
    "NIMAP": 34,  # unused variable
    "NISysABP": 35,  # unused variable
    "MechVent": 36,  # unused variable
    "RecordID": 37,  # unused variable
    "Age": 38,  # unused variable
    "Gender": 39,  # unused variable
    "ICUType": 40,  # unused variable
    "Height": 41  # unused variable
}


# functions to process the time in the data
def timeparser(time):
    return pd.to_timedelta(time + ':00')


def timedelta_to_day_figure(timedelta):
    return timedelta.days + (timedelta.seconds / 86400)  # (24*60*60) #add plural if Python 3


timesteps_list = []


def df_to_x_m_d(df, inputdict, size, id_posistion, split):
    grouped_data = df.groupby('Time')

    # generate input vectors
    x = np.zeros((len(inputdict) - 2, grouped_data.ngroups))
    masking = np.zeros((len(inputdict) - 2, grouped_data.ngroups))
    delta = np.zeros((split, size))
    timetable = np.zeros(grouped_data.ngroups)
    id = 0

    all_x = np.zeros((split, 1))

    s_dataset = np.zeros((3, split, size))

    if grouped_data.ngroups > size:

        # fill the x and masking vectors
        pre_time = pd.to_timedelta(0)
        t = 0
        for row_index, value in df.iterrows():
            '''
            t = colum, time frame
            agg_no = row, variable
            '''
            # print(value)
            agg_no = inputdict[value.Parameter]

            # same timeline check.
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                timetable[t] = timedelta_to_day_figure(value.Time)

            # print('agg_no : {}\t t : {}\t value : {}'.format(agg_no, t, value.Value))
            x[agg_no, t] = value.Value
            masking[agg_no, t] = 1

        '''
        # generate random index array 
        ran_index = np.random.choice(grouped_data.ngroups, size=size, replace=False)
        ran_index.sort()
        ran_index[0] = 0
        ran_index[size-1] = grouped_data.ngroups-1
        '''

        # generate index that has most parameters and first/last one.
        ran_index = grouped_data.count()
        ran_index = ran_index.reset_index()
        ran_index = ran_index.sort_values('Value', ascending=False)
        ran_index = ran_index[:size]
        ran_index = ran_index.sort_index()
        ran_index = np.asarray(ran_index.index.values)
        ran_index[0] = 0
        ran_index[size - 1] = grouped_data.ngroups - 1

        # print(ran_index)

        # take id for outcome comparing
        id = x[id_posistion, 0]

        timesteps_list.append(x.shape[1])

        # remove unnesserly parts(rows)
        x = x[:split, :]
        masking = masking[:split, :]

        # coulme(time) sampling
        x_sample = np.zeros((split, size))
        m_sample = np.zeros((split, size))
        time_sample = np.zeros(size)

        t_x_sample = x_sample.T
        t_marsking = m_sample.T
        # t_time = t_sample.T

        t_x = x.T
        t_m = masking.T
        # t_t = t.T

        it = np.nditer(ran_index, flags=['f_index'])
        while not it.finished:
            # print('it.index = {}, it[0] = {}, ran_index = {}'.format(it.index, it[0], ran_index[it.index]))
            t_x_sample[it.index] = t_x[it[0]]
            t_marsking[it.index] = t_m[it[0]]
            time_sample[it.index] = timetable[it[0]]
            it.iternext()

        x = x_sample
        masking = m_sample
        timetable = time_sample
        '''
        # normalize the X
        nor_x = x/max_input[:, np.newaxis]
        '''
        # fill the delta vectors
        for index, value in np.ndenumerate(masking):
            '''
            index[0] = row, agg
            index[1] = col, time
            '''
            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1] - 1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1] - 1] + delta[
                    index[0], index[1] - 1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1] - 1]

    else:

        # fill the x and masking vectors
        pre_time = pd.to_timedelta(0)
        t = 0
        for row_index, value in df.iterrows():
            '''
            t = colum, time frame
            agg_no = row, variable
            '''
            # print(value)
            agg_no = inputdict[value.Parameter]

            # same timeline check.
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                timetable[t] = timedelta_to_day_figure(value.Time)

            # print('agg_no : {}\t t : {}\t value : {}'.format(agg_no, t, value.Value))
            x[agg_no, t] = value.Value
            masking[agg_no, t] = 1

        # take id for outcome comparing
        id = x[id_posistion, 0]

        timesteps_list.append(x.shape[1])

        # remove unnesserly parts(rows)
        x = x[:split, :]
        masking = masking[:split, :]

        x = np.pad(x, ((0, 0), (size - grouped_data.ngroups, 0)), 'constant')
        masking = np.pad(masking, ((0, 0), (size - grouped_data.ngroups, 0)), 'constant')
        timetable = np.pad(timetable, (size - grouped_data.ngroups, 0), 'constant')
        '''
        # normalize the X
        nor_x = x/max_input[:, np.newaxis]
        '''
        # fill the delta vectors
        for index, value in np.ndenumerate(masking):
            '''
            index[0] = row, agg
            index[1] = col, time
            '''
            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1] - 1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1] - 1] + delta[
                    index[0], index[1] - 1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1] - 1]

    all_x = np.concatenate((all_x, x), axis=1)
    all_x = all_x[:, 1:]

    s_dataset[0] = x
    s_dataset[1] = masking
    s_dataset[2] = delta

    return s_dataset, all_x, id, timesteps_list


size = 49  # steps ~ from the paper
id_posistion = 37
input_length = 33  # input variables ~ from the paper
dataset = np.zeros((1, 3, input_length, size))

all_x_add = np.zeros((input_length, 1))

q = 0
with open('order.txt', 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        print(q, line)
        q += 1
        df = pd.read_csv(inputpath + line, header=0, parse_dates=['Time'], date_parser=timeparser)
        s_dataset, all_x, id, ts_list = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion,
                                                    split=input_length)

        dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
        all_x_add = np.concatenate((all_x_add, all_x), axis=1)

dataset = dataset[1:, :, :, :]
# (4000, 3, 33, 49)
ts_list = np.asarray(ts_list)
ts_list = np.reshape(ts_list, (1, -1))
print('ts_list shape ', ts_list.shape)
print('Dataset shape', dataset.shape)
print(dataset[0].shape)
print(dataset[0][0][0])
print(all_x_add.shape)
all_x_add = all_x_add[:, 1:]
print(all_x_add.shape)

trp = 0.64


def get_mean(x):
    x_mean = []
    l = int(trp * x.shape[1])
    for i in range(x.shape[0]):
        mean = np.mean(x[i][:l])
        x_mean.append(mean)
    return x_mean


def get_median(x):
    x_median = []
    l = int(trp * x.shape[1])
    for i in range(x.shape[0]):
        median = np.median(x[i][:l])
        x_median.append(median)
    return x_median


def get_std(x):
    x_std = []
    l = int(trp * x.shape[1])
    for i in range(x.shape[0]):
        std = np.std(x[i][:l])
        x_std.append(std)
    return x_std


def get_var(x):
    x_var = []
    l = int(trp * x.shape[1])
    for i in range(x.shape[0]):
        var = np.var(x[i][:l])
        x_var.append(var)
    return x_var


x_mean = get_mean(all_x_add)
print(x_mean)
print(len(x_mean))

x_std = get_std(all_x_add)
print(x_std)
print(len(x_std))


# dataset shape : (4000, 3, 33, 49)
def dataset_normalize(dataset, mean, std):
    for i in range(dataset.shape[0]):
        dataset[i][0] = (dataset[i][0] - mean[:, None])
        dataset[i][0] = dataset[i][0] / std[:, None]

    return dataset


x_mean = np.asarray(x_mean)
x_std = np.asarray(x_std)

# In[32]:

print('Normalizing dataset...')
dataset = dataset_normalize(dataset=dataset, mean=x_mean, std=x_std)
print(dataset[0][0][0])


# In[33]:


def normalize_chk(dataset):
    all_x_add = np.zeros((dataset[0][0].shape[0], 1))
    for i in range(dataset.shape[0]):
        all_x_add = np.concatenate((all_x_add, dataset[i][0]), axis=1)

    mean = get_mean(all_x_add)
    median = get_median(all_x_add)
    std = get_std(all_x_add)
    var = get_var(all_x_add)

    print('mean')
    print(mean)
    print('median')
    print(median)
    print('std')
    print(std)
    print('var')
    print(var)

    return mean, median, std, var


nor_mean, nor_median, nor_std, nor_var = normalize_chk(dataset)

print('Saving new dataset....')
np.save('./pre/dataset_' + str(size) + '_' + str(trp), dataset)
np.save('./pre/ts_lengths_' + str(size) + '_' + str(trp), ts_list)


# only check In-hospital_death
def df_to_y1(df):
    output = df.values
    output = output[:, 5:]

    return output


# only check In-hospital_death
def df_to_survival(df):
    output = np.zeros((4000, 1))

    for row_index, value in df.iterrows():
        sur = value[4]  # Survival
        if sur != -1:
            output[row_index] = 1
        else:
            output[row_index] = 0

    return output


A_outcomes = pd.read_csv('./Outcomes-a.txt')
y1_outcomes = df_to_y1(A_outcomes)
print(y1_outcomes.shape)
np.save('./pre/outcomes_' + str(size) + '_' + str(trp), y1_outcomes)
