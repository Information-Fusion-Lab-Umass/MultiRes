import cPickle as pickle


if __name__ == '__main__':
    data = pickle.load(open('/home/sidongzhang/code/fl/data/imputed_physionet.pkl', 'rb'))
    train_label = data['train']['label']
    print(train_label)