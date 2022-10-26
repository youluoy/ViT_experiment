#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import sys
sys.path.append("../modules")
sys.path.append("../settings")
import ViT_001 as ViT_001
import ViT_001_settings as settings
#モデルの設定
num_classes = settings.num_classes
batch_size = settings.batch_size
channel = settings.channel
height = settings.height
width = settings.width
emb_dim = settings.emb_dim
num_patch_row = settings.num_patch_row
num_blocks = settings.num_blocks
head = settings.head
hidden_dim = settings.hidden_dim
dropout = settings.dropout


#%%

#学習時ハイパラ、ネットワークの設定
lr = settings.lr
n_epochs = settings.n_epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ViT_001.Vit(in_channels=channel, num_classes=num_classes, num_patch_row=num_patch_row)
net.to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(net.parameters(), lr=lr) 



# ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train, valid用データのダウンロード
trainval_dataset = torchvision.datasets.CIFAR100(
    root='../data/inputs/cifar100/', 
    train=True, 
    download=True, 
    transform=transform
)
# train, validに分割
n_samples = len(trainval_dataset) # n_samples is 60000
train_size = int(len(trainval_dataset) * 0.8) # train_size is 48000
valid_size = n_samples - train_size # val_size is 48000
train_dataset, valid_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, valid_size])
# dataloaderを定義
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)
# test用データをダウンロード
test_dataset = torchvision.datasets.CIFAR100(
    root='../data/inputs/cifar100/', 
    train=False, 
    download=True, 
    transform=transform
)
# dataloaderを定義
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
)



#%%
# 学習
history_loss_train = []
history_acc_train  = []
history_loss_valid = []
history_acc_valid  = []

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    train_num = 0
    train_true_num = 0
    valid_num = 0
    valid_true_num = 0

    net.train()  #train
    for x, t in train_loader:

        # 勾配の初期化
        net.zero_grad()

        # targetをone-hot表現に
        #t_hot = torch.eye(10)[t]

        # テンソルをGPUに移動
        #x = x.reshape(x.shape[0], 1, x.shape[1])
        x = x.to(device)
        #t_hot = t_hot.to(device)
        t = t.to(device)
        #print(t.size())

        # 順伝播
        y = net(x)
        #y = test_net2.forward(x)
        #y = y.reshape(-1,)

        # 誤差の計算(BCE)
        #y = torch.clamp(y, 10^-15, 1-10^-15)
        #loss = -(t * torch.log(y) + (1 - t) * torch.log(1 - y)).mean()
        #print(y.size())
        loss = criterion(y, t) #y: one_hot, t: label_number

        # 誤差の逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # 予測をone-hotからラベルに戻す
        #y = y.reshape(-1,)
        pred = torch.argmax(y, 1)
        #pred = torch.where(y > 0.5, torch.ones_like(y), torch.zeros_like(y))

        losses_train.append(loss.tolist())

        acc = torch.where(t - pred == 0, torch.ones_like(t), torch.zeros_like(t))
        train_num += acc.size()[0]
        train_true_num += acc.sum().item()

    net.eval()  # 評価時eval
    for x, t in valid_loader:

        # targetをone-hot表現に
        #t_hot = torch.eye(10)[t]
        # テンソルをGPUに移動
        #x = x.reshape(x.shape[0], 1, x.shape[1])
        x = x.to(device)
        #t_hot = t_hot.to(device)
        t = t.to(device)

        # 順伝播
        y = net(x)
        #y = test_net2.forward(x)
        #y = y.reshape(-1,)

        # 誤差の計算(BCE)
        #loss = -(t * torch.log(y) + (1 - t) * torch.log(1 - y)).mean()
        loss = criterion(y, t)

        # the prediction is 
        #y = y.reshape(-1,)
        pred = torch.argmax(y,1)
        #pred = torch.where(y > 0.5, torch.ones_like(y), torch.zeros_like(y))

        losses_valid.append(loss.tolist())

        acc = torch.where(t - pred == 0, torch.ones_like(t), torch.zeros_like(t))
        valid_num += acc.size()[0]
        valid_true_num += acc.sum().item()

    #print(epoch)
    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}],  (train_num: {})'.format(
        epoch,
        np.mean(losses_train),
        train_true_num/train_num,
        np.mean(losses_valid),
        valid_true_num/valid_num,
        train_num
    ))
    history_loss_train.append(np.mean(losses_train))
    history_acc_train.append(train_true_num/train_num)
    history_loss_valid.append(np.mean(losses_valid))
    history_acc_valid.append(valid_true_num/valid_num)


    #if keyboard.is_pressed("s") and keyboard.is_pressed("t"):
    #    print("training was stopped by keyboard interrupt")
    #    break




# モデルの保存
#modelとhistoryの対応付けのためのid
import datetime
id_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

torch.save(net.state_dict(), "../data/outputs/trained_model/model_{}.pth".format(id_now)) 
#"../data/outputs/model_{}.pth".format(id_now)

#loss,accの保存
#history_loss_train, history_acc_train, history_loss_valid, history_acc_valid = [0.843,0.593,0.480,0.395,0.288], [0.522,0.690,0.755,0.822,0.851], [0.943,0.793,0.680,0.530,0.423], [0.480,0.650,0.705,0.802,0.801]

df = pd.DataFrame({'Train_loss': history_loss_train,
                   'Train_acc' : history_acc_train,
                   'Valid_loss': history_loss_valid,
                   'Valid_acc' : history_acc_valid
                    })
# CSV ファイル出力
df.to_csv("../data/outputs/history/history_{}.csv".format(id_now))
#csvの名前リスト取得
fnames = pd.read_csv('../data/outputs/history/filename_list.csv')
#新しい実行結果ののったcsvのfilenameをfilenameリストの最後に追加(consoleで実行後、下のコードでグラフを見れるようにしたい)
fnames = fnames.append({'filename': '../data/outputs/history/history_{}.csv'.format(id_now)}, ignore_index=True)
fnames.to_csv('../data/outputs/history/filename_list.csv', index=False)



#%%
















#++-----------------------------------
#過去のデータ閲覧用
#++-----------------------------------
#history.csvからのグラフ表示
import pandas as pd
import matplotlib.pyplot as plt
#hist_id = id_now
#hist_id = '2022-10-19-17-54-15'
#csvの名前リスト取得
fnames = pd.read_csv('../data/outputs/history/filename_list.csv')
#最新の結果のfilename取得
end_id = fnames.shape[0] - 1
lastfname = fnames.loc[end_id].values.tolist()[0]

#lastfname.csvのグラフ表示
histdata = pd.read_csv(lastfname, index_col=0)
histdata.plot(title = 'loss_history', y = ['Train_loss', 'Valid_loss'], colormap = 'prism')
plt.show()
histdata.plot(title = 'acc_history', y = ['Train_acc', 'Valid_acc'], colormap = 'prism')
plt.show()







