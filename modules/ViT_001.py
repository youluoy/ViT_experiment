# -*- coding: utf-8 -*-
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


#%%
# ----------------------------
# 2-3 Input Layer
# ----------------------------
print("=======2-3 Input Layer=======")

class VitInputLayer(nn.Module): 
    def __init__(self, in_channels:int=3, emb_dim:int=384, num_patch_row:int=2, image_size:int=32):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 高さ方向のパッチの数。例は2x2であるため、2をデフォルト値とした 
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
        """
        super(VitInputLayer, self).__init__() 
        self.in_channels=in_channels 
        self.emb_dim = emb_dim 
        self.num_patch_row = num_patch_row 
        self.image_size = image_size
        
        # パッチの数
        ## 例: 入力画像を2x2のパッチに分ける場合、num_patchは4 
        self.num_patch = self.num_patch_row**2

        # パッチの大きさ
        ## 例: 入力画像の1辺の大きさが32の場合、patch_sizeは16 
        self.patch_size = int(self.image_size // self.num_patch_row)

        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層 
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.emb_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        # クラストークン 
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim) 
        )

        # 位置埋め込み
        ## クラストークンが先頭に結合されているため、
        ## 長さemb_dimの位置埋め込みベクトルを(パッチ数+1)個用意 
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            x: 入力画像。形状は、(B, C, H, W)。[式(1)]
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            z_0: ViTへの入力。形状は、(B, N, D)。
                B:バッチサイズ、N:トークン数、D:埋め込みベクトルの長さ
        """
        # パッチの埋め込み & flatten [式(3)]
        ## パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P) 
        ## ここで、Pはパッチ1辺の大きさ
        z_0 = self.patch_emb_layer(x)

        ## パッチのflatten (B, D, H/P, W/P) -> (B, D, Np) 
        ## ここで、Npはパッチの数(=H*W/Pˆ2)
        z_0 = z_0.flatten(2)

        ## 軸の入れ替え (B, D, Np) -> (B, Np, D) 
        z_0 = z_0.transpose(1, 2)

        # パッチの埋め込みの先頭にクラストークンを結合 [式(4)] 
        ## (B, Np, D) -> (B, N, D)
        ## N = (Np + 1)であることに留意
        ## また、cls_tokenの形状は(1,1,D)であるため、
        ## repeatメソッドによって(B,1,D)に変換してからパッチの埋め込みとの結合を行う 
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)

        # 位置埋め込みの加算 [式(5)] 
        ## (B, N, D) -> (B, N, D) 
        z_0 = z_0 + self.pos_emb
        return z_0

batch_size, channel, height, width= 4, 3, 32, 32
x = torch.randn(batch_size, channel, height, width) 
input_layer = VitInputLayer(num_patch_row=2) 
z_0=input_layer(x)

# (2, 5, 384)(=(B, N, D))になっていることを確認。 
print(z_0.shape)


#%%
# ----------------------------
# 2-4 Self-Attention
# ----------------------------
print("=======2-4 Self-Attention=======")

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=3, dropout:float=0.):
        """ 
        引数:
            emb_dim: 埋め込み後のベクトルの長さ 
            head: ヘッドの数
            dropout: ドロップアウト率
        """
        super(MultiHeadSelfAttention, self).__init__() 
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5 # D_hの二乗根。qk^Tを割るための係数

        # 入力をq,k,vに埋め込むための線形層。 [式(6)] 
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # 式(7)にはないが、実装ではドロップアウト層も用いる 
        self.attn_drop = nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層。[式(10)]
        ## 式(10)にはないが、実装ではドロップアウト層も用いる 
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout) 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            z: MHSAへの入力。形状は、(B, N, D)。
                B: バッチサイズ、N:トークンの数、D:ベクトルの長さ
        返り値:
            out: MHSAの出力。形状は、(B, N, D)。[式(10)]
                B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ
        """

        batch_size, num_patch, _ = z.size()

        # 埋め込み [式(6)]
        ## (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q,k,vをヘッドに分ける [式(10)]
        ## まずベクトルをヘッドの個数(h)に分ける
        ## (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        ## Self-Attentionができるように、
        ## (バッチサイズ、ヘッド、トークン数、パッチのベクトル)の形に変更する 
        ## (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # 内積 [式(7)]
        ## (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)
        ## (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N) 
        dots = (q @ k_T) / self.sqrt_dh
        ## 列方向にソフトマックス関数        #sigmoidにしてみる
        #attn = F.softmax(dots, dim=-1)
        attn = F.sigmoid(dots)
        ## ドロップアウト
        attn = self.attn_drop(attn)
        # 加重和 [式(8)]
        ## (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h) 
        out = attn @ v
        ## (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        ## (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層 [式(10)]
        ## (B, N, D) -> (B, N, D) 
        out = self.w_o(out) 
        return out

mhsa = MultiHeadSelfAttention()
out = mhsa(z_0) #z_0は2-2節のz_0=input_layer(x)で、形状は(B, N, D)

# (2, 5, 384)(=(B, N, D))になっていることを確認 
print(out.shape)



# ----------------------------
# 2-5 Encoder
# ----------------------------
print("=======2-5 Encoder=======")

class VitEncoderBlock(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=8, hidden_dim:int=384*4, dropout: float=0.):
        """
        引数:
            emb_dim: 埋め込み後のベクトルの長さ
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ 
                        原論文に従ってemb_dimの4倍をデフォルト値としている
            dropout: ドロップアウト率
        """
        super(VitEncoderBlock, self).__init__()
        # 1つ目のLayer Normalization [2-5-2項]
        self.ln1 = nn.LayerNorm(emb_dim)
        # MHSA [2-4-7項]
        self.msa = MultiHeadSelfAttention(
        emb_dim=emb_dim, head=head,
        dropout = dropout,
        )
        # 2つ目のLayer Normalization [2-5-2項] 
        self.ln2 = nn.LayerNorm(emb_dim)
        # MLP [2-5-3項]
        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            z: Encoder Blockへの入力。形状は、(B, N, D)
                B: バッチサイズ、N:トークンの数、D:ベクトルの長さ
        返り値:
            out: Encoder Blockへの出力。形状は、(B, N, D)。[式(10)]
                B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ 
        """
        # Encoder Blockの前半部分 [式(12)] 
        out = self.msa(self.ln1(z)) + z
        # Encoder Blockの後半部分 [式(13)] 
        out = self.mlp(self.ln2(out)) + out 
        return out

vit_enc = VitEncoderBlock()
z_1 = vit_enc(z_0) #z_0は2-2節のz_0=input_layer(x)で、形状は(B, N, D)

# (2, 5, 384)(=(B, N, D))になっていることを確認 
print(z_1.shape)



# ----------------------------
# 2-6 ViTの実装
# ----------------------------
print("=======2-6 ViTの実装=======")

class Vit(nn.Module): 
    def __init__(self, in_channels:int=3, num_classes:int=10, emb_dim:int=384, num_patch_row:int=2, image_size:int=32, num_blocks:int=7, head:int=8, hidden_dim:int=384*4, dropout:float=0.):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            num_classes: 画像分類のクラス数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 1辺のパッチの数
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定 
            num_blocks: Encoder Blockの数
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ 
            dropout: ドロップアウト率
        """
        super(Vit, self).__init__()
        # Input Layer [2-3節] 
        self.input_layer = VitInputLayer(
            in_channels, 
            emb_dim, 
            num_patch_row, 
            image_size)

        # Encoder。Encoder Blockの多段。[2-5節] 
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout = dropout
            )
            for _ in range(num_blocks)])

        # MLP Head [2-6-1項] 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: ViTへの入力画像。形状は、(B, C, H, W)
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            out: ViTの出力。形状は、(B, M)。[式(10)]
                B:バッチサイズ、M:クラス数 
        """
        # Input Layer [式(14)]
        ## (B, C, H, W) -> (B, N, D)
        ## N: トークン数(=パッチの数+1), D: ベクトルの長さ 
        out = self.input_layer(x)
        
        # Encoder [式(15)、式(16)]
        ## (B, N, D) -> (B, N, D)
        out = self.encoder(out)

        # クラストークンのみ抜き出す
        ## (B, N, D) -> (B, D)
        cls_token = out[:,0]

        # MLP Head [式(17)]
        ## (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        return pred

num_classes = 100
batch_size, channel, height, width= 5000, 3, 32, 32
x = torch.randn(batch_size, channel, height, width)
vit = Vit(in_channels=channel, num_classes=num_classes) 
pred = vit(x)

# (2, 10)(=(B, M))になっていることを確認 
print(pred.shape)


#%%


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
#shape等の確認
#obj = test_dataset
#print()
#self.で定義されている変数を確認
#print(obj.__dict__.keys())
#print(obj.data.shape)

#print(train_loader.__dict__.keys())
#type(train_loader.dataset)
#obj = trainval_dataset
#print(obj.__dict__.keys())
#print(obj.class_to_idx)
#%%

# network
net = Vit(in_channels=channel, num_classes=num_classes) 
lr = 0.001
n_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"
net.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() 
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) 
optimizer = optim.Adam(net.parameters(), lr=lr) 

history_loss_train = []
history_acc_train  = []
history_loss_valid = []
history_acc_valid  = []

#%%

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



#%%
#モデルの保存

#modelとhistoryの対応付けのためのid
import datetime
id_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

#%%
torch.save(net.state_dict(), "../data/outputs/trained_model/model_{}.pth".format(id_now)) 
#"../data/outputs/model_{}.pth".format(id_now)


#%%
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










#%%
a = torch.ones(2, 3, 4, 4)
su = torch.sum(a, 1)
print(su.size())
print(su.unsqueeze(dim=1).size())


print(a.unfold(2, 2, 2).size())
print(a.unfold(3, 2, 2).size())


b = torch.ones(2, 4, 4, 4)

inputs = torch.ones(2,3,4,4)
filters = torch.ones(3,3,2,2)
F.conv2d(inputs, filters, padding=0)
filters[:,1,:,:] = torch.ones(2,2) + torch.ones(2,2)
filters[:,2,:,:] = torch.ones(2,2) + torch.ones(2,2) + torch.ones(2,2)
filters

c = F.conv2d(inputs, filters, padding=0)
print(c.size())
#print(c)
#%%
e = torch.zeros(2, 3, 4, 4)
r = 0
for i in range(2):
    for j in range(3):
        for k in range(4):
            for l in range(4):
                a[i,j,k,l] = r
                r += 1

d = e.unfold(2, 2, 2).unfold(3, 2, 2)
print(d.size())
#%%
f = d.reshape(2, 3, -1, 2, 2)
print(f.size())
print(f)
g = f.permute(0, 2, 1, 3, 4)
print(g.size())
print(g)

print(inputs.size())
img = inputs.unsqueeze(dim=1)
print(img.size())
print(g.size())
#%%
print("inputs.{}".format(inputs[0].unsqueeze(dim=0).size()))
window = g[0]
print("window.{}".format(window.size()))
h = F.conv2d(inputs[0].unsqueeze(dim=0), window, padding=0)
print("h.{}".format(h.size()))
print(h)
# %%
print(h.size()[0])
n = torch.zeros_like(h)
m = n.size()
#m[0] = 10
print(m)

hh = torch.zeros(1,2,2)
ww = torch.zeros(1,2,2)
bb = []
bb.append(hh)
bb.append(ww)
cc = torch.cat(bb, dim = 0)
print(cc.size())

#%%
torch.cuda.is_available()
x = torch.zeros(1, 2, 2)
y = x.copy()
y[:,:,:] = 1
print(x)

#%%
device = 'cuda'
x = torch.tensor([2.0], device=device, requires_grad=False)
w = torch.tensor([1.0], device=device, requires_grad=True)
b = torch.tensor([3.0], device=device, requires_grad=True)

y = x*w + b
y.backward()

z = w.detach()
s = z.clone()

print(z)
print(z.grad)
print(z is w)
print(z.is_leaf)

print(s)
print(s.grad)
print(s is z)
print(s.is_leaf)
#%%


aa = torch.ones(2, 2, 2, 2)
aa = torch.flatten(aa, start_dim=2)
aa.size()
m = nn.ReLU()
print(m(aa))

#%%
l = 0
cc = torch.zeros(2, 3, 4, 2, 2)
for i in range(cc.size()[0]):
    for j in range(cc.size()[1]):
        for k in range(cc.size()[2]):  
            cc[i,j,k,:] = l
            l += 1
#print(cc)
cc = cc.permute(0, 1, 3, 4, 2)
cc = cc.reshape(2, 3*2*2, 4)
fold = nn.Fold(output_size=(4, 4), kernel_size=(2,2))
cc = fold(cc)
print(cc)


#%%

l=0
cc = torch.zeros(2, 3, 4, 2, 2)
for i in range(cc.size()[0]):
    for j in range(cc.size()[1]):
        for k in range(cc.size()[2]):  
            cc[i,j,k,:] = l
            l += 1
cc = cc.permute(0, 2, 1, 3, 4)
dd = torchvision.utils.make_grid(cc[0], nrow=2, padding=0)
print(dd)
dd.size()


#%%
li = nn.Linear(10, 20)
ab = torch.ones(2, 3, 4, 10)
cd = li(ab)
print(cd.size())
