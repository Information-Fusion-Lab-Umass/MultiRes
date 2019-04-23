import torch

import cPickle as pickle


def validate_no_nans_in_tensor(tensor):
    assert not (tensor != tensor).any(), "null exists in input!"


if __name__ == '__main__':
    data = pickle.load(open('./data/3Sets_Physionet_avg_Covs_InHosp_set1.pkl'))
    d = (data['data']['141309'])
    print(d[0])
    print(d[1])
    print(d[2])
    print(d[3])
    print(d[4])