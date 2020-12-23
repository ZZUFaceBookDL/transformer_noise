import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset

class OzeDataset(Dataset):
    def __init__(self, filepath):
        super(OzeDataset, self).__init__()
        xy = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)
        # 此时取出来的数据是一个(N,M)的元组，N是行，M是列，xy.shape[0]可以把N取出来，也就是一共有多少行，有多少条数据
        # xy = torch.Tensor(xy)
        # result = torch.Tensor([])
        # for i in range(10):
        #     line = xy[i::10, :]
        #     result = torch.cat((result, line), dim=0)
        # xy = result
        # print(result)
        np.random.shuffle(xy)

        self.len = xy.shape[0]
        # self.x_data = torch.from_numpy(xy[:, 1:])
        self.x_data = xy[:, 1:]

        self.y_data = xy[:, 0]
        # self.y_data = torch.from_numpy(xy[:, 0])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_path = 'E:\\PyCharmWorkSpace\\别人的代码\\UCRArchive_2018\\ACSF1\\ACSF1_TRAIN.tsv'  # 数据集路径
# xy = np.loadtxt(train_path, delimiter='\t', dtype=np.float32)
#
# # print(xy)
# xy = torch.Tensor(xy)
# # xy = torch.chunk(xy, 10, dim=0)
# # xy = torch.cat(xy, dim=1)
# #
# # # print(xy.shape)
# result = torch.Tensor([]).float()
#
# # for line in xy:
# #     line = line.reshape(10, -1)
# #     line = torch.cat(torch.chunk(line, 10, dim=1), dim=0)
# #     print(line)
# #     # result.append(line)
# #
# # print(xy.shape)
# for i in range(10):
#     line = xy[i::10, :]
#     result = torch.cat((result, line), dim=0)

dataset_train = OzeDataset(train_path)

dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=100, shuffle=False)

x_data = y_data = None
for i in range(1):
    for x, y in dataloader_train:
        x_data = x
        y_data = y
    print('===================================')

x_data = x_data.unsqueeze(-1)
print(x_data)

x_data = x_data.expand(x_data.shape[0], x_data.shape[1], 3)
print(x_data.shape)