import cPickle as pickle

import numpy as np

import random


def get_easy(data_path):
    imputation = {}
    data = pickle.load(open(data_path, 'rb'))
    train_ids = data['train_ids']
    test_ids = data['test_ids']
    val_ids = data['val_ids']

    ids_dict = {'train': train_ids,
                'test': test_ids,
                'val': val_ids}

    for id_key in ids_dict.keys():
        all_imputations = []
        all_labels = []

        actual_lens = []

        for i in ids_dict[id_key]:
            d = data['data'][i]
            feats = np.asarray(d[0])
            flags = np.asarray(d[1])
            label = d[4]

            actual_len, num_features = flags.shape

            all_labels.append(label)
            actual_lens.append(actual_len)

            feats_im = []  # (seq_len, 37, 4)

            # print(actual_len, max_len)

            # for every 37 features
            for feat_ind in range(num_features):
                one_feats = []  # (seq_len, 4)
                feat = feats[:, feat_ind]  # historical data for ith feats
                feat_flag = flags[:, feat_ind]  # corresponding missing labels for ith feats
                ind_keep = feat_flag == 0
                if sum(ind_keep) > 0:  # if in the whole historical data there exists at least one missing point
                    avg_val = np.mean(feat[ind_keep])  # we calculate the mean feats value for ith feat
                else:
                    avg_val = 0.0
                last_val_observed = avg_val
                delta_t = -1
                for ind, each_flag in enumerate(feat_flag):
                    # we visit all the historical data point for ith feat
                    # and insert all the imputed historical data [feat_value, avg_value, is_missing, delta_time]
                    if each_flag == 1:
                        imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                        one_feats.append(imputation_feat)
                    elif each_flag == 0:
                        delta_t = 0
                        last_val_observed = feat[ind]
                        imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                        one_feats.append(np.array(imputation_feat))
                    delta_t += 1

                feats_im.append(one_feats)
            # feats_im = np.array(feats_im).transpose((1, 0, 2))
            # print(np.array(feats_im).shape)
            all_imputations.append(feats_im)
        imputation[id_key] = {'data': all_imputations, 'label': all_labels, 'lens': actual_lens}
        print(np.array(all_imputations).shape)
    return imputation


def get_imputation(data_path, max_len):
    """
    :param data_path: string; the loaded data has structure:
                      {'data':{id: [[value], [flags], [timestamp], label]},
                       'train_ids': [list of train_ids], 'test_ids': [list of test_ids], 'val_ids': [list of val_ids]}
                       where value and flags are [time_seq, 37] matrix
    :param num_features: int; the number of features; we will cluster those features later and do dot attention in each
                         cluster and among all clusters
    ;:return imputation_dict: dict; {'train': {'data':[[37, max_len, 4], [37, time_seq1, 4] ... ],
                                               'label': [label0, label1, ...]},
                                     'test': {'data': [[37, max_len, 4], [37, time_seq1, 4] ... ],
                                              'label': [label0, label1, ...]},
                                     'val': {'data': [[37, max_len, 4], [37, time_seq1,4] ... ],
                                             'label': [label0, label1, ...]}
                                     }
            lens: dict; {'train': [2560] vec,
                         'test': [] vec,
                         'val': [] vec
                         }
                              where each [[time_seq0, 37, 4], [time_seq1, 37 4] ... ] has len = number of data points
                             [[time_seq0, 37, 4], [time_seq1, 37, 4] ... ] cannot be transformed to a 4D tensor
                             for parallel computation, because time_seqs are different and a padded one does not support
                             DataParallel
    """
    imputation = {}
    data = pickle.load(open(data_path, 'rb'))
    train_ids = data['train_ids']
    test_ids = data['test_ids']
    val_ids = data['val_ids']

    ids_dict = {'train': train_ids,
                'test': test_ids,
                'val': val_ids}

    for id_key in ids_dict.keys():
        all_imputations = []
        all_labels = []

        actual_lens = []

        for i in ids_dict[id_key]:
            d = data['data'][i]
            feats = np.asarray(d[0])
            flags = np.asarray(d[1])
            label = d[3]

            actual_len, num_features = flags.shape

            all_labels.append(label)
            actual_lens.append(actual_len)

            feats_im = []  # (seq_len, 37, 4)

            # print(actual_len, max_len)

            # for every 37 features
            for feat_ind in range(num_features):
                one_feats = []  # (seq_len, 4)
                feat = feats[:, feat_ind]  # historical data for ith feats
                feat_flag = flags[:, feat_ind]  # corresponding missing labels for ith feats
                ind_keep = feat_flag == 0
                if sum(ind_keep) > 0:  # if in the whole historical data there exists at least one missing point
                    avg_val = np.mean(feat[ind_keep])  # we calculate the mean feats value for ith feat
                else:
                    avg_val = 0.0
                last_val_observed = avg_val
                delta_t = -1
                for ind, each_flag in enumerate(feat_flag):
                    # we visit all the historical data point for ith feat
                    # and insert all the imputed historical data [feat_value, avg_value, is_missing, delta_time]

                    # cut off
                    if ind >= max_len:
                        break

                    if each_flag == 1:
                        imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                        one_feats.append(imputation_feat)
                    elif each_flag == 0:
                        delta_t = 0
                        last_val_observed = feat[ind]
                        imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                        one_feats.append(np.array(imputation_feat))
                    delta_t += 1

                # padding
                for ind in range(actual_len, max_len):
                    one_feats.append(np.array([0, 0, 0, 0]))

                feats_im.append(one_feats)
            # feats_im = np.array(feats_im).transpose((1, 0, 2))
            # print(np.array(feats_im).shape)
            all_imputations.append(feats_im)
        imputation[id_key] = {'data': all_imputations, 'label': all_labels, 'lens': actual_lens}
        print(np.array(all_imputations).shape)
    return imputation


def get_origin_imputation(data_path):
    """
    :param data_path: string; the loaded data has structure:
                      {'data':{id: [[value], [flags], [timestamp], label]},
                       'train_ids': [list of train_ids], 'test_ids': [list of test_ids], 'val_ids': [list of val_ids]}
                       where value and flags are [time_seq, 37] matrix
    :param num_features: int; the number of features; we will cluster those features later and do dot attention in each
                         cluster and among all clusters
    ;:return imputation_dict: dict; {'train': {'data':[[37, max_len, 4], [37, time_seq1, 4] ... ],
                                               'label': [label0, label1, ...]},
                                     'test': {'data': [[37, max_len, 4], [37, time_seq1, 4] ... ],
                                              'label': [label0, label1, ...]},
                                     'val': {'data': [[37, max_len, 4], [37, time_seq1,4] ... ],
                                             'label': [label0, label1, ...]}
                                     }
            lens: dict; {'train': [2560] vec,
                         'test': [] vec,
                         'val': [] vec
                         }
                              where each [[time_seq0, 37, 4], [time_seq1, 37 4] ... ] has len = number of data points
                             [[time_seq0, 37, 4], [time_seq1, 37, 4] ... ] cannot be transformed to a 4D tensor
                             for parallel computation, because time_seqs are different and a padded one does not support
                             DataParallel
    """
    imputation = {}
    data = pickle.load(open(data_path, 'rb'))
    train_ids = data['train_ids']
    test_ids = data['test_ids']
    val_ids = data['val_ids']

    ids_dict = {'train': train_ids,
                'test': test_ids,
                'val': val_ids}

    for id_key in ids_dict.keys():
        all_imputations = []
        all_labels = []

        actual_lens = []

        for i in ids_dict[id_key]:
            d = data['data'][i]
            feats = np.asarray(d[0])
            flags = np.asarray(d[1])
            label = d[3]

            actual_len, num_features = flags.shape

            all_labels.append(label)
            actual_lens.append(actual_len)

            feats_im = []  # (seq_len, 37, 4)

            # print(actual_len, max_len)

            # for every 37 features
            for feat_ind in range(num_features):
                one_feats = []  # (seq_len, 4)
                feat = feats[:, feat_ind]  # historical data for ith feats
                feat_flag = flags[:, feat_ind]  # corresponding missing labels for ith feats
                ind_keep = feat_flag == 0
                if sum(ind_keep) > 0:  # if in the whole historical data there exists at least one missing point
                    avg_val = np.mean(feat[ind_keep])  # we calculate the mean feats value for ith feat
                else:
                    avg_val = 0.0
                last_val_observed = avg_val
                delta_t = -1
                for ind, each_flag in enumerate(feat_flag):
                    # we visit all the historical data point for ith feat
                    # and insert all the imputed historical data [feat_value, avg_value, is_missing, delta_time]
                    if each_flag == 1:
                        imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                        one_feats.append(imputation_feat)
                    elif each_flag == 0:
                        delta_t = 0
                        last_val_observed = feat[ind]
                        imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                        one_feats.append(np.array(imputation_feat))
                    delta_t += 1

                feats_im.append(one_feats)
            # feats_im = np.array(feats_im).transpose((1, 0, 2))
            # print(np.array(feats_im).shape)
            all_imputations.append(feats_im)
        imputation[id_key] = {'data': all_imputations, 'label': all_labels, 'lens': actual_lens}
        print(np.array(all_imputations).shape)
    return imputation


def get_forward_imputation(data_path):
    imputation = {}
    data = pickle.load(open(data_path, 'rb'))
    train_ids = data['train_ids']
    test_ids = data['test_ids']
    val_ids = data['val_ids']

    ids_dict = {'train': train_ids,
                'test': test_ids,
                'val': val_ids}

    for id_key in ids_dict.keys():
        all_imputations = []
        all_labels = []

        actual_lens = []

        for i in ids_dict[id_key]:
            d = data['data'][i]
            feats = np.asarray(d[0])
            flags = np.asarray(d[1])
            label = d[3]

            actual_len, num_features = flags.shape

            all_labels.append(label)
            actual_lens.append(actual_len)

            feats_im = []  # (seq_len, 37, 4)

            # print(actual_len, max_len)

            # for every 37 features
            for feat_ind in range(num_features):
                one_feats = []  # (seq_len, 4)
                feat = feats[:, feat_ind]  # historical data for ith feats
                feat_flag = flags[:, feat_ind]  # corresponding missing labels for ith feats
                ind_keep = feat_flag == 0
                if sum(ind_keep) > 0:  # if in the whole historical data there exists at least one missing point
                    avg_val = np.mean(feat[ind_keep])  # we calculate the mean feats value for ith feat
                else:
                    avg_val = 0.0
                last_val_observed = avg_val
                delta_t = -1
                for ind, each_flag in enumerate(feat_flag):
                    # we visit all the historical data point for ith feat
                    # and insert all the imputed historical data [feat_value, avg_value, is_missing, delta_time]
                    if each_flag == 1:
                        # imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                        imputation_feat = last_val_observed
                        one_feats.append(imputation_feat)
                    elif each_flag == 0:
                        delta_t = 0
                        last_val_observed = feat[ind]
                        # imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                        imputation_feat = last_val_observed
                        one_feats.append(imputation_feat)
                    delta_t += 1

                feats_im.append(one_feats)
            feats_im = np.array(feats_im).transpose((1, 0))
            feats_im = feats_im.tolist()
            # print(np.array(feats_im).shape)
            all_imputations.append(feats_im)
        imputation[id_key] = {'data': all_imputations, 'label': all_labels, 'lens': actual_lens}
        print(np.array(all_imputations).shape)
    return imputation


def get_tbm_imputation(data_path):
    imputation = {}
    data = pickle.load(open(data_path, 'rb'))
    train_ids = data['train_ids']
    test_ids = data['test_ids']
    val_ids = data['val_ids']

    ids_dict = {'train': train_ids,
                'test': test_ids,
                'val': val_ids}

    for id_key in ids_dict.keys():
        all_imputations = []  # (B, 37, T)
        all_labels = []

        actual_lens = []

        for i in ids_dict[id_key]:
            d = data['data'][i]
            feats = np.asarray(d[0])
            flags = np.asarray(d[1])
            label = d[4]

            actual_len, num_features = flags.shape

            all_labels.append(label)
            actual_lens.append(actual_len)

            feats_im = []  # (37, seq_len, 4)

            # print(actual_len, max_len)

            # for every 37 features
            for feat_ind in range(num_features):
                one_feats = []  # (seq_len, 4)
                feat = feats[:, feat_ind]  # historical data for ith feats
                feat_flag = flags[:, feat_ind]  # corresponding missing labels for ith feats
                ind_keep = feat_flag == 0
                if sum(ind_keep) > 0:  # if in the whole historical data there exists at least one missing point
                    avg_val = np.mean(feat[ind_keep])  # we calculate the mean feats value for ith feat
                else:
                    avg_val = 0.0
                last_val_observed = avg_val
                delta_t = -1
                for ind, each_flag in enumerate(feat_flag):
                    # we visit all the historical data point for ith feat
                    # and insert all the imputed historical data [feat_value, avg_value, is_missing, delta_time]
                    if each_flag == 1:
                        imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                        one_feats.append(imputation_feat)
                    elif each_flag == 0:
                        delta_t = 0
                        last_val_observed = feat[ind]
                        imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                        one_feats.append(np.array(imputation_feat))
                    delta_t += 1

                feats_im.append(one_feats)
            # print(np.array(feats_im).shape)
            local_b = []  # (37, T)
            for f in range(num_features):
                one_feat = feats_im[f]
                local_f = []
                for t in range(len(one_feat)):
                    curr_feat = one_feat[t]
                    # if(curr_feat[2]==1):
                    # TBM parameters
                    beta_val = 0.75
                    tau_val = 2
                    h_val = 0.4
                    m_t = curr_feat[2]
                    x_l = curr_feat[0]
                    x_m = curr_feat[1]
                    curr_delta_t = curr_feat[3]
                    b_t_dash = np.exp(-beta_val * curr_delta_t * 1.0 / tau_val)
                    if b_t_dash > h_val:
                        b_t = 1
                    else:
                        b_t = 0
                    feat_val = (1 - m_t) * x_l + m_t * (b_t * x_l + (1 - b_t) * x_m)
                    local_f.append(feat_val)
                local_b.append(local_f)
            all_imputations.append(local_b)
            #
            # raw_tbm = np.array(raw_tbm)  # (B, F, T)
            # raw_tbm = np.transpose(raw_tbm, (0, 2, 1))  # (B, T, F)
        imputation[id_key] = {'data': all_imputations, 'label': all_labels, 'lens': actual_lens}

        print(np.array(all_imputations).shape)

    return imputation



def small_train_data(data_path, ids):
    imputated = pickle.load(open(data_path, 'rb'))
    data = []
    label = []
    lens = []
    traindata = imputated['train']['data']
    trainlabel = imputated['train']['label']
    trainlens = imputated['train']['label']
    for i in ids:
        data.append(traindata[i])
        label.append(trainlabel[i])
        lens.append(trainlens[i])
    print(np.array(data).shape)
    small_train = {'data': data, 'label': label, 'lens': lens}
    pickle.dump(small_train, open('./data/small_train.pkl', 'wb'))


if __name__ == '__main__':
    # data = get_imputation('./data/final_Physionet_avg_new.pkl', 116)
    # pickle.dump(data, open('./data/pc_physionet.pkl', 'wb'))

    # data = get_origin_imputation('./final_Physionet_avg_new.pkl')
    # pickle.dump(data, open('./data/origin_physionet.pkl', 'wb'))

    # ids = random.sample(range(0, 2517), 50)
    # small_train_data('./data/pc_physionet.pkl', ids)

    # print('data1')
    # data = get_origin_imputation('./data/phy_data_set_1.pkl')
    # pickle.dump(data, open('./data/origin_physionet_1.pkl', 'wb'))
    # print
    #
    # print('data2')
    # data = get_origin_imputation('./data/phy_data_set_2.pkl')
    # pickle.dump(data, open('./data/origin_physionet_2.pkl', 'wb'))
    # print
    #
    # print('data3')
    # data = get_origin_imputation('./data/phy_data_set_3.pkl')
    # pickle.dump(data, open('./data/origin_physionet_3.pkl', 'wb'))
    # print

    # print('data1')
    # data = get_forward_imputation('./data/phy_data_set_1.pkl')
    # pickle.dump(data, open('./data/forward_physionet_1.pkl', 'wb'))
    # print
    #
    # print('data2')
    # data = get_forward_imputation('./data/phy_data_set_2.pkl')
    # pickle.dump(data, open('./data/forward_physionet_2.pkl', 'wb'))
    # print
    #
    # print('data3')
    # data = get_forward_imputation('./data/phy_data_set_3.pkl')
    # pickle.dump(data, open('./data/forward_physionet_3.pkl', 'wb'))
    # print

    # print('data1')
    # data = get_tbm_imputation('./data/phy_data_set_1.pkl')
    # pickle.dump(data, open('./data/tbm_physionet_1.pkl', 'wb'))
    # print
    #
    # print('data2')
    # data = get_tbm_imputation('./data/phy_data_set_2.pkl')
    # pickle.dump(data, open('./data/tbm_physionet_2.pkl', 'wb'))
    # print
    #
    # print('data3')
    # data = get_tbm_imputation('./data/phy_data_set_3.pkl')
    # pickle.dump(data, open('./data/tbm_physionet_3.pkl', 'wb'))
    # print

    print('data1')
    data = get_tbm_imputation('./data/3Sets_Physionet_avg_Covs_InHosp_set1.pkl')
    pickle.dump(data, open('./data/easy_physionet_1.pkl', 'wb'))
    print

    print('data2')
    data = get_tbm_imputation('./data/3Sets_Physionet_avg_Covs_InHosp_set2.pkl')
    pickle.dump(data, open('./data/easy_physionet_2.pkl', 'wb'))
    print

    print('data3')
    data = get_tbm_imputation('./data/3Sets_Physionet_avg_Covs_InHosp_set3.pkl')
    pickle.dump(data, open('./data/easy_physionet_3.pkl', 'wb'))
    print


