import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

import torchvision.models as models
import sys

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        # model = models.resnet50(pretrained=True)
        model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)

        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = ResNetFeatureExtractor().to(device)
for param in feature_extractor.parameters():
    param.requires_grad = False


def compute_feature_difference(real, fake, feature_extractor):
    real_features = feature_extractor(real)
    fake_features = feature_extractor(fake)
    return F.mse_loss(real_features, fake_features)


# 需要判别器D的损失持续上升而G生成器的损失持续下降
# 加载数据集并应用数据增强
class CustomDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.data_info = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        img_name = os.path.join(self.data_dir, self.data_info.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_info.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data_info)

def load_dataset(csv_path, data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256,256)),  #
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset(csv_path, data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader


class ResidualBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockTranspose, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        test2=self.main(x) + self.shortcut(x)
        return F.relu(test2)

# 定义生成器网络（简化版的DCGAN生成器）

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),  # 修改这里: 更多的初始通道
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            ResidualBlockTranspose(ngf * 32, ngf * 16, stride=2),  # 新增的块
            ResidualBlockTranspose(ngf * 16, ngf * 8, stride=2),
            ResidualBlockTranspose(ngf * 8, ngf * 4, stride=2),
            ResidualBlockTranspose(ngf * 4, ngf * 2, stride=2),
            ResidualBlockTranspose(ngf * 2, ngf, stride=2)
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.initial(input)
        x = self.main(x)
        x = self.final(x)
        return x




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        test=self.main(x) + self.shortcut(x)
        return F.relu(test)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            # I've removed a few residual blocks to simplify the discriminator
            ResidualBlock(ndf, ndf * 2, stride=2),
            nn.LeakyReLU(0.2, inplace=False),
            # nn.Dropout(0.3),

            ResidualBlock(ndf * 2, ndf * 4, stride=2),
            nn.LeakyReLU(0.2, inplace=False),
            # nn.Dropout(0.3),


            ResidualBlock(ndf * 4, ndf * 8, stride=2),
            nn.LeakyReLU(0.2, inplace=False),
            # nn.Dropout(0.3),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层
        self.fc = nn.Sequential(
            nn.Linear(ndf * 8, 1),  # Changed from ndf * 16 to ndf * 8
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

        

def train_dcgan(data_loader, generator, discriminator, criterion, g_optimizer, d_optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generator.to(device)
    discriminator.to(device)
    min_diff = float('inf') # 初始化为无穷大
    
    g_losses = []
    d_losses = []

    # writer = SummaryWriter('runs/experiment_name')

    fixed_noise = torch.randn(32, nz, 1, 1, device=device)  # 用于保存固定的随机噪声以便可视化生成效果

    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(data_loader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            target_real = torch.full((batch_size, 1), 1.0, device=device)
            target_fake = torch.full((batch_size, 1), 0.0, device=device)

            # 训练生成器
            for idx, (generator, g_optimizer) in enumerate(zip(generators, g_optimizers)):
                generator.to(device)
                generator.train()
                generator.zero_grad()

                # 生成假数据
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_data = generator(noise)
                output_fake = discriminator(fake_data)
                g_loss = criterion(output_fake, target_real)
                g_loss.backward()

                g_optimizer.step()

                # 每5次生成器的迭代后训练一次判别器
                if (idx + 1) % 5 == 0:
                    # 训练判别器
                    discriminator.train() 
                    d_optimizer.zero_grad()

                    # 训练真实样本
                    output_real = discriminator(real_data)
                    d_loss_real = criterion(output_real, target_real)
                    d_loss_real.backward()

                    # 使用最后一个生成器生成假数据来训练判别器
                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake_data = generators[-1](noise)  
                    output_fake = discriminator(fake_data.detach())
                    d_loss_fake = criterion(output_fake, target_fake)
                    d_loss_fake.backward()

                    d_optimizer.step()

                    # 记录生成器和判别器的损失
                    # writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(data_loader) + i)
                    # writer.add_scalar('Loss/Discriminator', (d_loss_real + d_loss_fake).item(), epoch * len(data_loader) + i)

                    g_losses.append(g_loss.item())
                    d_losses.append((d_loss_real + d_loss_fake).item())

                    if i % 100 == 0:
                        print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(data_loader)} Loss_D: {d_loss_real + d_loss_fake:.4f} Loss_G: {g_loss:.4f}")

        # 每个Epoch结束后，可视化生成效果并保存生成的图片
        if epoch>300:
            generator.eval()
            # noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # fake_data = generator(fixed_noise)
            # diff = compute_feature_difference(real_data, fake_data, feature_extractor)
            for generator_idx, generator in enumerate(generators):
                generator.eval()
                fake_data = generator(fixed_noise)
                num_batches_to_save = 5
                for i in range(num_batches_to_save):
                    # noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake_data = generator(fixed_noise)
                    diff = compute_feature_difference(real_data, fake_data, feature_extractor)
                    # save_generated_images(fake_data, epoch)


                # 如果当前生成的图像与真实图像的差异是迄今为止的最小值，则保存它
                    if diff.item() < min_diff:
                        min_diff = diff.item()
                        save_generated_images(fake_data, epoch, i)

                    else:
                        min_diff = diff.item()
                        save_generated_images(fake_data, epoch, i+1000)
          
        
        # with torch.no_grad():
        #     fake_images = generator(fixed_noise).detach().cpu()
        #     save_generated_images(fake_images, epoch)


#     # 使用matplotlib绘制损失曲线
#     plt.figure(figsize=(10, 5))
#     plt.title("Generator and Discriminator Loss During Training")
#     plt.plot(g_losses, label="G")
#     plt.plot(d_losses, label="D")
#     plt.xlabel("iterations")
#     plt.ylabel("Loss")
#     plt.legend()

#     # 保存图像
#     save_path = "loss_curve.png"
#     plt.savefig(save_path)

#     # 展示图像
#     plt.show()

#     print(f"Loss curve saved to {save_path}")


        if epoch%200==0:
    # 使用matplotlib绘制损失曲线
            save=1
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(g_losses, label="G")
            plt.plot(d_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()

            # 生成唯一的文件名
            base_name = "loss_curve"
            extension = ".png"
            save_path = get_unique_filename(base_name, extension)

            # 保存图像
            plt.savefig(save_path)

            # 展示图像
            plt.show()

            print(f"Loss curve saved to {save_path}")
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    # 生成唯一的文件名
    base_name = "loss_curve"
    extension = ".png"
    save_path = get_unique_filename(base_name, extension)

    # 保存图像
    plt.savefig(save_path)

    # 展示图像
    plt.show()

    print(f"Loss curve finish saved to {save_path}")

    
    
def get_unique_filename(base_name, extension):
    version = 1
    while os.path.exists(f"{base_name}_{version}{extension}"):
        version += 1
    return f"{base_name}_{version}{extension}"

def save_generated_images(fake_output, epoch, number):
    # base_name = "loss_curve"
    # extension = ".png"
    # get_unique_filename(generated_images)
    root_dir = "autodl-tmp/mydata/generated_images_class0_v1"
    # 创建一个基于epoch的子文件夹
    save_dir = os.path.join(root_dir, f"Epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    for idx, image in enumerate(fake_output):
        image = (image + 1) / 2  # 将图像的像素值从[-1, 1]范围转换到[0, 1]范围
        image = transforms.ToPILImage()(image)
        image.save(os.path.join(save_dir, f"Epoch_{epoch}_Number_{number}_Step_{idx}.png"))
    
# 主函数
if __name__ == "__main__":
    csv_path = "autodl-tmp/mydata/train_full_data_class0.csv"
    data_dir = "autodl-tmp/mydata/0"
    data_loader = load_dataset(csv_path, data_dir, batch_size=32)

    nz = 128  # 潜在向量的维度
    ngf = 64  # 生成器的特征图大小
    nc = 3    # 图像通道数（对于彩色图像，通道数为3）
    ndf = 64  # 判别器的特征图大小
    
#     generator = Generator(nz, ngf, nc)
#     discriminator = Discriminator(ndf, nc)
    criterion = nn.BCELoss()
#     g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))
    
    
    # 创建生成器和判别器列表
    num_generators = 5
    generators = [Generator(nz, ngf, nc) for _ in range(num_generators)]
    discriminator = Discriminator(ndf, nc)

    # 创建生成器的优化器列表
    g_optimizers = [optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) for generator in generators]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))

    # g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5) #添加了L2正则化
    # d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999),  weight_decay=1e-5) # L2正则化

    train_dcgan(data_loader, generators, discriminator, criterion, g_optimizers, d_optimizer, num_epochs=1000)
