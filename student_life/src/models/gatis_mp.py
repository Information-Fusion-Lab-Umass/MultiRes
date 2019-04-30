import sys, os
import torch
import torch.nn as nn

from src.bin import validations
''' 
Location Features (8 total):
 - total distance covered in a day
 - maximum 2-point displacement in a day
 - distance standard deviatio
 - number of different areas visited by tiles approximation
 - total spatial coverage by convex hull,
 - difference in sequence of tiles covered compared to previous day
 - difference in sequence of clusters visited compared to previous day
 - distance entropy

4 Temporal one-hot features 
- weekends, start of term, mid-term, end of term

'''
class GatisNet(torch.nn.Module):
    def __init__(self, n_feature=12, hidden_dims=[57, 35, 35 ,3]):
        super(GatisNet, self).__init__()
        self.n_feature          = n_feature
        self.hidden_dims        = hidden_dims 
        self.dense              = nn.Sequential(
            nn.Linear(n_feature, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.Softmax()
        )


    def forward(self, features):
        logits = self.dense(features)
        return logits

if __name__ == '__main__':
    print("test")