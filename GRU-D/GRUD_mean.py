#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numbers
import warnings

import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils

# In[2]:


inputpath = '../cluster_attn/set-a/'
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


# In[3]:


# functions to process the time in the data
def timeparser(time):
    return pd.to_timedelta(time + ':00')


def timedelta_to_day_figure(timedelta):
    print(timedelta, timedelta.days, timedelta.seconds)
    return timedelta.days + (timedelta.seconds / 86400)  # (24*60*60)


# In[4]:


# group the data by time
def df_to_inputs(df, inputdict, inputs):
    grouped_data = df.groupby('Time')

    for row_index, value in df.iterrows():
        '''
        t = colum ~ time frame
        agg_no = row ~ variable
        '''

        agg_no = inputdict[value.Parameter]

        # print('agg_no : {}\t  value : {}'.format(agg_no, value.Value))
        inputs[agg_no].append(value.Value)

    return inputs


# In[5]:


inputs = []

# prepare empty list to put data
# len(inputdict)-2: two items has same agg_no
# for i in range(len(inputdict)-2):
#    t = []
#    inputs.append(t)

q = 0
# read all the files in the input folder
# for filename in os.listdir(inputpath):
# with open('order.txt','r') as f:
#   for line in f:
#       #print(line)
#       line = line.rstrip('\n')
#       print(q, line)
#       q+=1
#       df = pd.read_csv(inputpath + line, header=0, parse_dates=['Time'], date_parser=timeparser) 
#       inputs = df_to_inputs(df=df, inputdict=inputdict, inputs=inputs)

# print(inputs[0][0])


# In[6]:


# save inputs just in case
# np.save('./input/inputs_oursplit', inputs)
# inputs = np.load('./input/inputs_oursplit.npy')
# print(inputs[0][0])


# In[7]:


# make input items list
input_columns = list(inputdict.keys())

'''
remove two overlaped items
"TroponinI" : 27, #temp
"TroponinT" : 28, #temp

'''
input_columns.remove("TroponinI")
input_columns.remove("TroponinT")
print(input_columns)
print(len(input_columns))


# In[8]:


# describe the data
# print count, min, max, mean, median, std, var and histogram if hist == True
# return values as a list
def describe(inputs, input_columns, inputdict, hist=False):
    desc = []

    for i in range(len(inputdict) - 2):
        input_arr = np.asarray(inputs[i])

        des = []

        des.append(input_arr.size)
        des.append(np.amin(input_arr))
        des.append(np.amax(input_arr))
        des.append(np.mean(input_arr))
        des.append(np.median(input_arr))
        des.append(np.std(input_arr))
        des.append(np.var(input_arr))

        desc.append(des)

        # print histgram
        if hist:
            a = np.hstack(input_arr)
            plt.hist(a, bins='auto')
            plt.title("Histogram about {}".format(input_columns[i]))
            plt.show()

        print('count: {}, min: {}, max: {}'.format(des[0], des[1], des[2]))
        print('mean: {}, median: {}, std: {}, var: {}'.format(des[3], des[4], des[5], des[6]))

    return desc


# In[9]:
# print('Describing ...')

# desc = describe(loaded_inputs, input_columns, inputdict, hist=True)
# desc = np.asarray(desc)
# desc.shape


# In[10]:


# save desc
# 0: count, 1: min, 2: max, 3: mean, 4: median, 5: std, 6: var
# np.save('./input/desc', desc)
# loaded_desc = np.load('./input/desc.npy')


# In[11]:


def normalization(desc, inputs):
    # for each catagory
    for i in range(desc.shape[0]):
        # for each value
        for j in range(len(inputs[i])):
            inputs[i][j] = (inputs[i][j] - desc[i][3]) / desc[i][5]
    return inputs


# In[22]:

# print('Dataframe to dataset....')
'''
dataframe to dataset
'''


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

    return s_dataset, all_x, id


# In[23]:


# def df_to_x_m_d(df, inputdict, mean, std, size, id_posistion, split):
size = 49  # steps ~ from the paper
id_posistion = 37
input_length = 33  # input variables ~ from the paper
dataset = np.zeros((1, 3, input_length, size))

all_x_add = np.zeros((input_length, 1))

q = 0
# for filename in os.listdir(inputpath):
with open('../cluster_attn/order.txt', 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        print(q, line)
        q += 1
        df = pd.read_csv(inputpath + line, header=0, parse_dates=['Time'], date_parser=timeparser)
        s_dataset, all_x, id = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion,
                                           split=input_length)

        dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
        all_x_add = np.concatenate((all_x_add, all_x), axis=1)

dataset = dataset[1:, :, :, :]
# (total datasets, kind of data(x, masking, and delta), input length, num of varience)
# (4000, 3, 33, 49)
print(dataset.shape)
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


# In[25]:


def get_median(x):
    x_median = []
    l = int(trp * x.shape[1])
    for i in range(x.shape[0]):
        median = np.median(x[i][:l])
        x_median.append(median)
    return x_median


# In[26]:


def get_std(x):
    x_std = []
    l = int(trp * x.shape[1])
    for i in range(x.shape[0]):
        std = np.std(x[i][:l])
        x_std.append(std)
    return x_std


# In[27]:


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

# In[29]:


x_std = get_std(all_x_add)
print(x_std)
print(len(x_std))


# In[30]:


# dataset shape : (4000, 3, 33, 49)
def dataset_normalize(dataset, mean, std):
    for i in range(dataset.shape[0]):
        dataset[i][0] = (dataset[i][0] - mean[:, None])
        dataset[i][0] = dataset[i][0] / std[:, None]

    return dataset


# In[31]:


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


# In[34]:


nor_mean, nor_median, nor_std, nor_var = normalize_chk(dataset)

# In[35]:

print('Saving new dataset....')
np.save('./input/x_mean_aft_nor_70split_33', nor_mean)
np.save('./input/x_median_aft_nor_70split_33', nor_median)
np.save('./input/dataset_70split_33', dataset)
# print('Loading new dataset 36......')
# t_dataset = np.load('./input/dataset.npy')

# print(t_dataset.shape)


# In[26]:


'''
Y values
'''


def df_to_y3(df):
    '''
    RecordID  SAPS-I  SOFA  Length_of_stay  Survival  In-hospital_death
    '''
    output = np.zeros((4000, 3))

    for row_index, value in df.iterrows():
        los = value[3]  # Length_of_stay
        sur = value[4]  # Survival
        ihd = value[5]  # In-hospital_death

        output[row_index][0] = ihd
        output[row_index][1] = ihd

        # length-of-stay less than 3 yes/no 1/0
        if los < 3:
            output[row_index][2] = 0
        else:
            output[row_index][2] = 1

    return output


# In[27]:


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


# In[28]:


# A_outcomes = pd.read_csv('./input/temp/Outcomes-a.txt')
y1_outcomes = df_to_y1(A_outcomes)
# y1_outcomes = np.load('./input/y1_out.npy')
print(y1_outcomes.shape)
np.save('./input/y1_out', y1_outcomes)


# print('Getting survival outcomes.....')
# A_outcomes = pd.read_csv('./input/Outcomes-a.txt')
# survival_outcomes = df_to_survival(A_outcomes)
# print(survival_outcomes)
# np.save('./input/survival_out', survival_outcomes)

# In[29]:


def df_to_y2(df):
    '''
    RecordID  SAPS-I  SOFA  Length_of_stay  Survival  In-hospital_death
    '''
    output = np.zeros((4000, 2))

    for row_index, value in df.iterrows():
        ihd = value[5]  # In-hospital_death

        output[row_index][0] = ihd
        output[row_index][1] = ihd

    return output


# In[30]:


# A_outcomes = pd.read_csv('./input/Outcomes-a.txt')
# y2_outcomes = df_to_y2(A_outcomes)
# y2_outcomes = np.load('./input/y2_out.npy')
# print(y2_outcomes.shape)
# np.save('./input/y2_out', y2_outcomes)


# In[4]:


# define model
class GRUD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0, bias=True, batch_first=False,
                 bidirectional=False, dropout_type='mloss', dropout=0):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.zeros = torch.autograd.Variable(torch.zeros(input_size))
        self.x_mean = torch.autograd.Variable(torch.tensor(x_mean))
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        ################################
        gate_size = 1  # not used
        ################################

        self._all_weights = []

        '''
        w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        '''
        # decay rates gamma
        w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        # z
        w_xz = torch.nn.Parameter(torch.Tensor(input_size))
        w_hz = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mz = torch.nn.Parameter(torch.Tensor(input_size))

        # r
        w_xr = torch.nn.Parameter(torch.Tensor(input_size))
        w_hr = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mr = torch.nn.Parameter(torch.Tensor(input_size))

        # h_tilde
        w_xh = torch.nn.Parameter(torch.Tensor(input_size))
        w_hh = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mh = torch.nn.Parameter(torch.Tensor(input_size))

        # y (output)
        w_hy = torch.nn.Parameter(torch.Tensor(output_size, hidden_size))

        # bias
        b_dg_x = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_y = torch.nn.Parameter(torch.Tensor(output_size))

        layer_params = (
            w_dg_x, w_dg_h, w_xz, w_hz, w_mz, w_xr, w_hr, w_mr, w_xh, w_hh, w_mh, w_hy, b_dg_x, b_dg_h, b_z, b_r, b_h,
            b_y)

        param_names = ['weight_dg_x', 'weight_dg_h', 'weight_xz', 'weight_hz', 'weight_mz', 'weight_xr', 'weight_hr',
                       'weight_mr', 'weight_xh', 'weight_hh', 'weight_mh', 'weight_hy']
        if bias:
            param_names += ['bias_dg_x', 'bias_dg_h', 'bias_z', 'bias_r', 'bias_h', 'bias_y']

        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """
        Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(GRUD, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(GRUD, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []

        weights = ['weight_dg_x', 'weight_dg_h', 'weight_xz', 'weight_hz', 'weight_mz', 'weight_xr', 'weight_hr',
                   'weight_mr', 'weight_xh', 'weight_hh', 'weight_mh', 'weight_hy', 'bias_dg_x', 'bias_dg_h', 'bias_z',
                   'bias_r', 'bias_h', 'bias_y']

        if self.bias:
            self._all_weights += [weights]
        else:
            self._all_weights += [weights[:2]]

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def forward(self, input):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        X = torch.squeeze(input[0])  # .size = (33,49)
        Mask = torch.squeeze(input[1])  # .size = (33,49)
        Delta = torch.squeeze(input[2])  # .size = (33,49)
        Hidden_State = torch.autograd.Variable(torch.zeros(input_size))

        step_size = X.size(1)  # 49
        # print('step size : ', step_size)

        output = None
        h = Hidden_State

        # decay rates gamma
        w_dg_x = getattr(self, 'weight_dg_x')
        w_dg_h = getattr(self, 'weight_dg_h')

        # z
        w_xz = getattr(self, 'weight_xz')
        w_hz = getattr(self, 'weight_hz')
        w_mz = getattr(self, 'weight_mz')

        # r
        w_xr = getattr(self, 'weight_xr')
        w_hr = getattr(self, 'weight_hr')
        w_mr = getattr(self, 'weight_mr')

        # h_tilde
        w_xh = getattr(self, 'weight_xh')
        w_hh = getattr(self, 'weight_hh')
        w_mh = getattr(self, 'weight_mh')

        # bias
        b_dg_x = getattr(self, 'bias_dg_x')
        b_dg_h = getattr(self, 'bias_dg_h')
        b_z = getattr(self, 'bias_z')
        b_r = getattr(self, 'bias_r')
        b_h = getattr(self, 'bias_h')

        for layer in range(num_layers):

            x = torch.squeeze(X[:, layer:layer + 1])
            m = torch.squeeze(Mask[:, layer:layer + 1])
            d = torch.squeeze(Delta[:, layer:layer + 1])

            # (4)
            gamma_x = torch.exp(-torch.max(self.zeros, (w_dg_x * d + b_dg_x)))
            gamma_h = torch.exp(-torch.max(self.zeros, (w_dg_h * d + b_dg_h)))

            # (5)
            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.x_mean)

            # (6)
            if self.dropout == 0:
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))

                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''

                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = dropout(h_tilde)

                h = (1 - z) * h + z * h_tilde

            else:
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

        w_hy = getattr(self, 'weight_hy')
        b_y = getattr(self, 'bias_y')

        output = torch.matmul(w_hy, h) + b_y
        output = torch.sigmoid(output)

        return output


# In[5]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[6]:


def data_dataloader(dataset, outcomes, train_proportion=0.64, dev_proportion=0.2, test_proportion=0.16):
    train_index = int(np.floor(dataset.shape[0] * train_proportion))
    dev_index = int(np.floor(dataset.shape[0] * (train_proportion + dev_proportion)))

    # split dataset to tarin/dev/test set
    train_data, train_label = dataset[:train_index, :, :, :], outcomes[:train_index, :]
    test_data, test_label = dataset[train_index:dev_index, :, :, :], outcomes[train_index:dev_index, :]
    dev_data, dev_label = dataset[dev_index:, :, :, :], outcomes[dev_index:, :]

    # ndarray to tensor
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    # tensor to dataset
    train_dataset = utils.TensorDataset(train_data, train_label)
    dev_dataset = utils.TensorDataset(dev_data, dev_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    # dataset to dataloader 
    train_dataloader = utils.DataLoader(train_dataset)
    dev_dataloader = utils.DataLoader(dev_dataset)
    test_dataloader = utils.DataLoader(test_dataset)

    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))

    return train_dataloader, dev_dataloader, test_dataloader


# In[7]:


t_dataset = np.load('./input/dataset_70split_33.npy')
# t_dataset_old = np.load('./input/dataset.npy')
# print(t_dataset_old[0][0][0])
# t_out = np.load('./input/y1_out.npy')
# outcomes_oursplit = []
# with open('outcomes.txt', 'r') as f:
#     for line in f:
#         line = line.rstrip('\n')
#         outcomes_oursplit.append(int(line))
print(outcomes_oursplit)
t_out = np.asarray(outcomes_oursplit)
t_out.resize((4000, 1))
# t_out = np.load('./input/survival_out.npy')
print(t_dataset[0][0][0])
print(t_out[0])
print(t_dataset.shape)
print(t_out.shape)

train_dataloader, dev_dataloader, test_dataloader = data_dataloader(t_dataset, t_out)

# In[8]:


'''
in the paper : 49 layers, 33 input, 18838 parameters
input : 10-weights(*input), 6 - biases
Y: 1 weight(hidden*output), 1 bias(output)
Input : hidden : output : layer  = # of parameters : len(para)
1:1:1:1 = 18 : 18
2:1:1:1 = 25 : 18  // +7 as expected
1:1:1:2 = 34 : 18 // 34 = 16*2 + 2
33:33:1:1 = 562 : 18 // 16*33(528) + 33*1 +1 = 562
33:33:5:1 = 698 : 18 // 16*33(528) + 33*5(165) +5 = 698
33:33:5:49 = 26042 : 18 // 16*33*49(25872) + 33*5(165) +5 = 698
weights = 10*33*49(16170) + 33*5(165) = 16335 gap : 2503

'''


# In[8]:


def fit(model, criterion, learning_rate, train_dataloader, dev_dataloader, test_dataloader, learning_rate_decay=0,
        n_epochs=30):
    epoch_losses = []

    # to check the update 
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()

    graph_epochs = []
    fscores_train = []
    fscores_val = []
    fscores_test = []
    for epoch in range(n_epochs):

        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if epoch % learning_rate_decay == 0:
                learning_rate = learning_rate / 2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train the model
        losses, acc = [], []
        label, pred = [], []
        y_pred_col = []
        model.train()
        for train_data, train_label in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            train_data = torch.squeeze(train_data)
            train_label = torch.squeeze(train_label)

            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(train_data)

            # y_pred = y_pred[:, None]
            # train_label = train_label[:, None]

            # print(y_pred.shape)
            # print(train_label.shape)

            # Save predict and label
            y_pred_col.append(y_pred.item())
            pred.append(y_pred.item())
            label.append(train_label.item())

            # print('y_pred: {}\t label: {}'.format(y_pred, train_label))

            # Compute loss
            loss = criterion(y_pred, train_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.item())

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)

        pred = np.asarray(pred)
        label = np.asarray(label)

        train_pred_out = pred
        train_label_out = label

        train_p2 = pred
        train_p2[train_p2 >= 0.5] = 1
        train_p2[train_p2 < 0.5] = 0

        # save new params
        new_state_dict = {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()

        # compare params
        for key in old_state_dict:
            if (old_state_dict[key] == new_state_dict[key]).all():
                print('Not updated in {}'.format(key))

        # dev loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for dev_data, dev_label in dev_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            dev_data = torch.squeeze(dev_data)
            dev_label = torch.squeeze(dev_label)

            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(dev_data)

            # Save predict and label
            pred.append(y_pred.item())
            label.append(dev_label.item())

            # Compute loss
            loss = criterion(y_pred, dev_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    dev_label)
            )
            losses.append(loss.item())

        dev_acc = torch.mean(torch.cat(acc).float())
        dev_loss = np.mean(losses)

        pred = np.asarray(pred)
        label = np.asarray(label)

        dev_pred_out = pred
        dev_label_out = label

        dev_p2 = pred
        dev_p2[dev_p2 >= 0.5] = 1
        dev_p2[dev_p2 < 0.5] = 0

        # test loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label)

            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(test_data)

            # Save predict and label
            pred.append(y_pred.item())
            label.append(test_label.item())

            # Compute loss
            loss = criterion(y_pred, test_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    test_label)
            )
            losses.append(loss.item())

        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)

        test_pred_out = pred
        test_label_out = label

        epoch_losses.append([
            train_loss, dev_loss, test_loss,
            train_acc, dev_acc, test_acc,
            train_pred_out, dev_pred_out, test_pred_out,
            train_label_out, dev_label_out, test_label_out,
        ])

        pred = np.asarray(pred)
        label = np.asarray(label)
        # print(train_acc, dev_acc, test_acc)
        # print(pred, label)
        auc_score = roc_auc_score(label, pred)
        test_p2 = pred
        test_p2[test_p2 >= 0.5] = 1
        test_p2[test_p2 < 0.5] = 0
        # print(p2, label)
        # print(precision_recall_fscore_support(label,p2,average='weighted'))

        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, dev_loss, dev_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Test loss: {:.4f}, Test AUC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, auc_score))
        prf_train = precision_recall_fscore_support(train_label_out, train_p2, average='weighted')
        prf_val = precision_recall_fscore_support(dev_label_out, dev_p2, average='weighted')
        prf_test = precision_recall_fscore_support(test_label_out, test_p2, average='weighted')

        print("TRAIN PRF: " + str(prf_train))
        print("VAL PRF: " + str(prf_val))
        print("TEST PRF: " + str(prf_test))

        fscores_train.append(prf_train[2])
        fscores_val.append(prf_val[2])
        fscores_test.append(prf_test[2])
        graph_epochs.append(epoch)

        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        # Train
        plt.plot(graph_epochs, fscores_train, color='red', label='Train')
        # Val
        plt.plot(graph_epochs, fscores_val, color='blue', label='Val')
        # Test
        plt.plot(graph_epochs, fscores_test, color='green', label='Test')

        plt.xlabel("Epochs")
        plt.ylabel('F-score')
        plt.title('GRU-D Physionet')
        plt.xticks([])
        ax.legend()
        plt.savefig(str(epoch) + "_GRUD.jpg")
        plt.show()
        # save the parameters
        train_log = []
        train_log.append(model.state_dict())
        torch.save(model.state_dict(), './save/grud_mean_grud_para.pt')

        # print(train_log)

    return epoch_losses


# In[9]:


def plot_roc_and_auc_score(outputs, labels, title):
    false_positive_rate, true_positive_rate, threshold = roc_curve(labels, outputs)
    auc_score = roc_auc_score(labels, outputs)
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve, AREA = {:.4f}'.format(auc_score))
    plt.plot([0, 1], [0, 1], 'red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


# In[10]:


input_size = 33  # num of variables base on the paper
hidden_size = 33  # same as inputsize
output_size = 1
num_layers = 49  # num of step or layers base on the paper

x_mean = torch.Tensor(np.load('./input/x_mean_aft_nor_70split_33.npy'))
x_median = torch.Tensor(np.load('./input/x_median_aft_nor_70split_33.npy'))

# In[23]:


# dropout_type : Moon, Gal, mloss
model = GRUD(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=0, dropout_type='mloss',
             x_mean=x_mean, num_layers=num_layers)

# load the parameters
# model.load_state_dict(torch.load('./save/grud_para.pt'))
# model.eval()

count = count_parameters(model)
print('number of parameters : ', count)
print(list(model.parameters())[0].grad)

criterion = torch.nn.BCELoss()

# In[24]:


'''
def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay=0, n_epochs=30):
'''
learning_rate = 0.01
learning_rate_decay = 7
n_epochs = 18

# learning_rate = 0.1 learning_rate_decay=True
epoch_losses = fit(model, criterion, learning_rate, train_dataloader, dev_dataloader, test_dataloader,
                   learning_rate_decay, n_epochs)

# In[25]:


test_preds, test_labels = epoch_losses[11][8], epoch_losses[11][11]

plot_roc_and_auc_score(test_preds, test_labels, 'GRU-D PhysioNet mortality prediction')

# In[ ]:
