import cPickle as pickle


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

    data1 = pickle.load(open('./data/phy_data_set_1.pkl', 'rb'))
    print(data1['data']['141784'])