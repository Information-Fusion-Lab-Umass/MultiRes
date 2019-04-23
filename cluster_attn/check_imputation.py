import cPickle as pickle

import numpy as np
import pandas as pd


if __name__ == '__main__':
    # data1 = pickle.load(open('./data/phy_data_set_1_split_threeway.pkl', 'rb'))
    # print(data1['train_ids'])
    # print(data1['test_ids'])
    # print(data1['val_ids'])
    # print('train size: ', len(data1['train_ids']))
    # print('test size: ', len(data1['test_ids']))
    # print('val size: ', len(data1['val_ids']))
    # print
    #
    # data2 = pickle.load(open('./data/phy_data_set_2_split_threeway.pkl', 'rb'))
    # print(data2['train_ids'])
    # print(data2['test_ids'])
    # print(data2['val_ids'])
    # print('train size: ', len(data2['train_ids']))
    # print('test size: ', len(data2['test_ids']))
    # print('val size: ', len(data2['val_ids']))
    # print
    #
    # data3 = pickle.load(open('./data/phy_data_set_3_split_threeway.pkl', 'rb'))
    # print(data3['train_ids'])
    # print(data3['test_ids'])
    # print(data3['val_ids'])
    # print('train size: ', len(data3['train_ids']))
    # print('test size: ', len(data3['test_ids']))
    # print('val size: ', len(data3['val_ids']))
    # print
    #
    # print(len(set(data1['train_ids']) & set(data2['train_ids'])))
    # print(len(set(data1['test_ids']) & set(data2['test_ids'])))
    # print(len(set(data1['val_ids']) & set(data2['val_ids'])))

    # data1 = pickle.load(open('./data/final_Physionet_avg_new.pkl', 'rb'))
    # clusters = [[4, 7], [8, 17, 29], [26, 34, 13], [20, 21, 22], [24, 36]]
    # count = 0
    # datapoints = set([])
    # for i in data1['data'].keys():
    #     val = np.array(data1['data'][i][0])  # (time_seq, 37)
    #     T = val.shape[0]
    #     for c in clusters:
    #         if len(c) == 2:
    #             if np.array_equal(val[:, c[0]], val[:, c[1]]):
    #                 datapoints.add(i)
    #                 print('data ' + i + ' has same values in %d and %d feats' % (c[0], c[1]))
    #                 if not np.array_equal(val[:, c[0]], np.zeros(T)):
    #                     print(val[:, c[0]])
    #         if len(c) == 3:
    #             if np.array_equal(val[:, c[0]], val[:, c[1]]):
    #                 datapoints.add(i)
    #                 print('data ' + i + ' has same values in %d and %d feats' % (c[0], c[1]))
    #                 if not np.array_equal(val[:, c[0]], np.zeros(T)):
    #                     print(val[:, c[0]])
    #             if np.array_equal(val[:, c[0]], val[:, c[2]]):
    #                 datapoints.add(i)
    #                 print('data ' + i + ' has same values in %d and %d feats' % (c[0], c[2]))
    #                 if not np.array_equal(val[:, c[0]], np.zeros(T)):
    #                     print(val[:, c[0]])
    #             if np.array_equal(val[:, c[1]], val[:, c[2]]):
    #                 datapoints.add(i)
    #                 print('data ' + i + ' has same values in %d and %d feats' % (c[1], c[2]))
    #                 if not np.array_equal(val[:, c[1]], np.zeros(T)):
    #                     print(val[:, c[1]])
    # print(len(datapoints))
    # pickle.dump(datapoints, open('dump_points.pkl', 'wb'))

    # data1 = pickle.load(open('./data/3Sets_Physionet_avg_Covs_InHosp_set1.pkl', 'rb'))
    # # df = pd.read_pickle('./data/3Sets_Physionet_avg_Covs_InHosp_set1.pkl')
    # # print(df)
    # print(data1.keys())
    # print(len(data1['data']['141290']))
    # print(data1['train_ids'])
    # print(data1['test_ids'])
    # print(data1['val_ids'])
    # mx = np.array(data1['data']['138525'][0])
    # flag = np.array(data1['data']['138525'][1])
    # timestamps = np.array(data1['data']['138525'][2])
    #
    # print(mx)
    # print(mx.shape)
    # print(flag)
    # print(flag.shape)
    # print(timestamps)
    # print(data1['data']['138525'][3])
    #
    # print(data1['data']['138525'])
    data = pickle.load(open('./data/physio_final_prf_CVL-Phy-.pkl', 'rb'))
    print(data)
