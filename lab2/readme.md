## Lab 2

### Part1 MNIST

### Part2 Debiasing

#### 3种偏见类型

1. Human Bias ： 基于数据的，来源于人类本身的偏见
2. Interaction Bias:  给出“画鞋子”的要求，给出的交互是带有偏见的
3. Latent Bias:  过去的科学家 大多数 男性
4. Selection Bias： 数据的选择有关

使用的数据集：

1. **Positive training data**: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). A large-scale (over 200K images) of celebrity faces.
2. **Negative training data**: [ImageNet](http://www.image-net.org/). Many images across many different categories. We'll take negative examples from a variety of non-human categories. [Fitzpatrick Scale](https://en.wikipedia.org/wiki/Fitzpatrick_scale) skin type classification system, with each image labeled as "Lighter'' or "Darker''.


虽然数据集是有标签的，但是模型学习到的特征可能有偏差。 目标： 构建一个在CelebA数据集上训练，并在测试数据集上对所有人群都达到高分类准确率的模型，证明模型不存在隐藏的偏差。



偏差： 需要考虑潜在变量， 如果分类器在感知到潜在特征后变化了分类决策， 认为分类器存在偏差

**baseline:** CNN模型， 卷积层和两个全连接层

```
Sequential(
  (0): ConvBlock(
    (conv): Conv2d(3, 12, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    (relu): ReLU(inplace=True)
    (bn): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): ConvBlock(
    (conv): Conv2d(12, 24, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    (relu): ReLU(inplace=True)
    (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (2): ConvBlock(
    (conv): Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (relu): ReLU(inplace=True)
    (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (3): ConvBlock(
    (conv): Conv2d(48, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (relu): ReLU(inplace=True)
    (bn): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (4): Flatten(start_dim=1, end_dim=-1)
  (5): Linear(in_features=1152, out_features=512, bias=True)
  (6): ReLU(inplace=True)
  (7): Linear(in_features=512, out_features=1, bias=True)
)
```

训练CNN模型:

```python
### Train the standard CNN ###
loss_fn = nn.BCEWithLogitsLoss()
# Training hyperparameters
params = dict(
    batch_size=32,
    num_epochs=2,  # keep small to run faster
    learning_rate=5e-4,
)

experiment = create_experiment("6S191_Lab2_Part2_CNN", params)

optimizer = optim.Adam(
    standard_classifier.parameters(), lr=params["learning_rate"]
)  # define our optimizer
loss_history = mdl.util.LossHistory(smoothing_factor=0.99)  # to record loss evolution
plotter = mdl.util.PeriodicPlotter(sec=2, scale="semilogy")
if hasattr(tqdm, "_instances"):
    tqdm._instances.clear()  # clear if it exists

# set the model to train mode
standard_classifier.train()


def standard_train_step(x, y):
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    # clear the gradients
    optimizer.zero_grad()

    # feed the images into the model
    logits = standard_classifier(x)
    # Compute the loss
    loss = loss_fn(logits, y)

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss


# The training loop!
step = 0
for epoch in range(params["num_epochs"]):
    for idx in tqdm(range(loader.get_train_size() // params["batch_size"])):
        # Grab a batch of training data and propagate through the network
        x, y = loader.get_batch(params["batch_size"])
        loss = standard_train_step(x, y)
        loss_value = loss.detach().cpu().numpy()

        # Record the loss and plot the evolution of the loss as a function of training
        loss_history.append(loss_value)
        plotter.plot(loss_history.get())

        experiment.log_metric("loss", loss_value, step=step)
        step += 1

experiment.end()
```

训练之后，对CNN的表现进行评估：

```python
### Evaluation of standard CNN ###

# set the model to eval mode
standard_classifier.eval()

# TRAINING DATA
# Evaluate on a subset of CelebA+Imagenet
(batch_x, batch_y) = loader.get_batch(5000)
batch_x = torch.from_numpy(batch_x).float().to(device)
batch_y = torch.from_numpy(batch_y).float().to(device)

with torch.inference_mode():
    y_pred_logits = standard_classifier(batch_x)
    # sigmoid将预测结果压缩到（0,1）区间， round四舍五入为二分类
    y_pred_standard = torch.round(torch.sigmoid(y_pred_logits))
    acc_standard = torch.mean((batch_y == y_pred_standard).float())

print(
    "Standard CNN accuracy on (potentially biased) training set: {:.4f}".format(
        acc_standard.item()
    )
)
```

在未见过且分类的数据集上：

<img width="706" height="430" alt="image" src="https://github.com/user-attachments/assets/97442b11-034f-4f5a-a699-8b452152eb78" />


由于训练集中大多数是Light Female， 所以训练出来的分类器更擅长识别和分类类似特征的面孔，从而产生偏差

克服偏差： 训练数据中标注不同的子类，保持平衡

缺点：1. 需要标注海量数据 2. 要求人事先知道应该寻找哪些潜在的偏差（种族？性别？）



所以需要无偏，无监督（unbiased, unsupervised）的方式学习特征，然后根据特征训练分类器

#### VAE (变分自编码器):

**使用VAE（Variational Autoencoders）以完全无监督的方式学习人脸数据中潜在特征的编码**
<img width="706" height="430" alt="image" src="https://github.com/user-attachments/assets/7cb113cb-35b7-447d-8a07-85d072373f03" />


##### VAE的两个关键方面：

###### 1. Loss Function 损失函数: 

学习潜在空间的时候我们是将均值和标准差约束为近似服从单位高斯分布，**这些是学习到的参数**， 所以也要纳入损失函数的计算。

所以我们的VAE损失函数包含：

- **潜在损失**：学习到的潜在变量与单位高斯分布的匹配程度， KL散度定义
  $$
  L_{KL}(\mu, \sigma) = \frac{1}{2}\sum_{j=0}^{k-1} (\sigma_j + \mu_j^2 - 1 - \log{\sigma_j})
  $$

- **重构损失：** 重构的输出与输入的匹配程度，由范数给出
  $$
  L_{x}{(x,\hat{x})} = ||x-\hat{x}||_1
  $$

所以VAE的损失函数：
$$
L_{VAE} = c\cdot L_{KL} + L_{x}{(x,\hat{x})}
$$
c是用于正则化的权重系数



代码表示：

```python
# TODO: Define the reconstruction loss as the mean absolute pixel-wise
# difference between the input and reconstruction. Hint: you'll need to
# use torch.mean, and specify the dimensions to reduce over.
# For example, reconstruction loss needs to average
# over the height, width, and channel image dimensions.
# https://pytorch.org/docs/stable/generated/torch.mean.html
reconstruction_loss = torch.mean(torch.abs(x - x_recon))# TODO

# TODO: Define the VAE loss. Note this is given in the equation for L_{VAE}
# in the text block directly above
vae_loss = kl_weight * latent_loss + reconstruction_loss # TODO

return vae_loss
```

2. ###### Reparameterization 重参数化 （trick）：

如果直接从分布中采样，采样本身是随机的，不可微。

VAE生成的是一个均值向量和一个标准差（方差的平方根）向量， 然后我们从标准差中采样并加上均值，输出采样后的潜在向量。因为约束为单位高斯分布，所以我们可以得到对于潜变量z：
$$
z = \mu + e^{\left(\frac{1}{2} \cdot \log{\Sigma}\right)}\circ \epsilon
$$
其中 $\mu$代表均值， $\Sigma$代表协方差矩阵（各个对角元素为各潜变量的方差），$\epsilon$是固定的随机噪声， 这样允许梯度通过z回传到均值和标准差上

```python
### VAE Reparameterization ###

"""Reparameterization trick by sampling from an isotropic unit Gaussian.
# Arguments
    z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
# Returns
    z (tensor): sampled latent vector
"""
def sampling(z_mean, z_logsigma):
    # Generate random noise with the same shape as z_mean, sampled from a standard normal distribution (mean=0, std=1)
    eps = torch.randn_like(z_mean)

    # # TODO: Define the reparameterization computation!
    # # Note the equation is given in the text block immediately above.
    z = z_mean + torch.exp(z_logsigma)*eps
    # TODO

    return z
```

#### 去偏变分自编码器（DB-VAE）

##### DB-VAE Model

用VAE学习到的潜在变量， 在训练过程中对训练集进行重采样。 较少见特征的人脸在训练过程中被采样的概率会更高

![DB-VAE](https://camo.githubusercontent.com/ab94df56f8b95f7e6f16230f8dc230857ed5d19334e0827792a2df47629db5c3/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4d4954446565704c6561726e696e672f696e74726f746f646565706c6561726e696e672f323031392f6c6162322f696d672f44422d5641452e706e67)

DB-VAE还会输出一个监督变量$z_o$对应类别预测：人脸or非人脸

对于人脸，DB-VAE即学习了无监督潜在向量的表示 $q_\phi(z|x)$ 又输出了监督类别预测$z_0$； 对于负样本，只负责输出类别预测

##### DB-VAE 损失函数

对于人脸;

1. VAE 损失： 同上，潜在损失和重构损失
2. 分类损失：二分类问题的Standard cross-entropy loss 

对于非人脸： 只有分类损失
$$
L_{total} = L_y(y,\hat{y}) + {I}_f(y)\Big[L_{VAE}\Big]
$$
代码实现：

```python
### Loss function for DB-VAE ###

"""Loss function for DB-VAE.
# Arguments
    x: true input x
    x_pred: reconstructed x
    y: true label (face or not face)
    y_logit: predicted labels
    mu: mean of latent distribution (Q(z|X))
    logsigma: log of standard deviation of latent distribution (Q(z|X))
# Returns
    total_loss: DB-VAE total loss
    classification_loss = DB-VAE classification loss

"""
def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    # TODO: call the relevant function to obtain VAE loss
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma) # TODO

    # TODO: define the classification loss using binary_cross_entropy
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_logit, y)# TODO

    # Use the training data labels to create variable face_indicator:
    #   indicator that reflects which training data are images of faces
    y = y.float()
    face_indicator = (y == 1.0).float()

    # TODO: define the DB-VAE total loss! Use torch.mean to average over all
    # samples
    total_loss = torch.mean( classification_loss + face_indicator * vae_loss )# TODO

    return total_loss, classification_loss### Loss function for DB-VAE ###

```

##### DB-VAE结构：

使用标准的CNN分类器作为编码器， 然后定义一个解码器网络（接受采样的隐变量，经过一系列反卷积层）。 端到端的VAE

```python
### Define the decoder portion of the DB-VAE ###

n_filters = 12  # base number of convolutional filters, same as standard CNN
latent_dim = 100  # number of latent variables


def make_face_decoder_network(latent_dim=100, n_filters=12):
    """
    Function builds a face-decoder network.

    Args:
        latent_dim (int): the dimension of the latent representation
        n_filters (int): base number of convolutional filters

    Returns:
        decoder_model (nn.Module): the decoder network
    """

    class FaceDecoder(nn.Module):
        def __init__(self, latent_dim, n_filters):
            super(FaceDecoder, self).__init__()

            self.latent_dim = latent_dim
            self.n_filters = n_filters

            # Linear (fully connected) layer to project from latent space
            # to a 4 x 4 feature map with (6*n_filters) channels
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 4 * 4 * 6 * n_filters), nn.ReLU()
            )

            # Convolutional upsampling (inverse of an encoder)
            self.deconv = nn.Sequential(
                # [B, 6n_filters, 4, 4] -> [B, 4n_filters, 8, 8]
                nn.ConvTranspose2d(
                    in_channels=6 * n_filters,
                    out_channels=4 * n_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
                # [B, 4n_filters, 8, 8] -> [B, 2n_filters, 16, 16]
                nn.ConvTranspose2d(
                    in_channels=4 * n_filters,
                    out_channels=2 * n_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
                # [B, 2n_filters, 16, 16] -> [B, n_filters, 32, 32]
                nn.ConvTranspose2d(
                    in_channels=2 * n_filters,
                    out_channels=n_filters,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.ReLU(),
                # [B, n_filters, 32, 32] -> [B, 3, 64, 64]
                nn.ConvTranspose2d(
                    in_channels=n_filters,
                    out_channels=3,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
            )

        def forward(self, z):
            """
            Forward pass of the decoder.

            Args:
                z (Tensor): Latent codes of shape [batch_size, latent_dim].

            Returns:
                Tensor of shape [batch_size, 3, 64, 64], representing
                the reconstructed images.
            """
            x = self.linear(z)  # [B, 4*4*6*n_filters]
            x = x.view(-1, 6 * self.n_filters, 4, 4)  # [B, 6n_filters, 4, 4]

            # Upsample through transposed convolutions
            x = self.deconv(x)  # [B, 3, 64, 64]
            return x

    return FaceDecoder(latent_dim, n_filters
```

整体的DB-VAE结构：

```python
### Defining and creating the DB-VAE ###


class DB_VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim

        # Define the number of outputs for the encoder.
        self.encoder = make_standard_classifier(n_outputs=2 * latent_dim + 1)
        self.decoder = make_face_decoder_network()

    # function to feed images into encoder, encode the latent space, and output
    def encode(self, x):
        encoder_output = self.encoder(x)

        # classification prediction
        y_logit = encoder_output[:, 0].unsqueeze(-1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1 : self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1 :]

        return y_logit, z_mean, z_logsigma

    # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        # TODO: call the sampling function defined above
        z = sampling(z_mean, z_logsigma)# TODO
        return z

    # Decode the latent space and output reconstruction
    def decode(self, z):
        # TODO: use the decoder to output the reconstruction
        reconstruction = self.decoder(z)# TODO
        return reconstruction

    # The forward function will be used to pass inputs x through the core VAE
    def forward(self, x):
        # Encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)

        # TODO: reparameterization
        z = self.reparameterize(z_mean,z_logsigma)# TODO

        # TODO: reconstruction
        recon = self.decode(z)# TODO

        return y_logit, z_mean, z_logsigma, recon

    # Predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit

dbvae = DB_VAE(latent_dim)
```

之后再进行训练和评估：
<img width="759" height="426" alt="image" src="https://github.com/user-attachments/assets/4384e2d4-3805-4fc6-92aa-870414cb4b64" />



