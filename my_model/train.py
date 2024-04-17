import math
import time

import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split

from model.fcos import FCOSDetector
from AircraftDataset import AircraftDataset

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# 获取训练集
train_dataset = AircraftDataset("../dataset", mode='train', resize_size=[512, 800])
print("success creating train_dataset...")

# 模型和优化器
model = FCOSDetector(mode="train").to(device)
model.load_state_dict(torch.load("./weights/aircraft_512x800_epoch23_loss0.5937.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("success loading model...")
# 训练参数:批量大小、训练轮数和学习率warm-up的步数占总步数的比例
BATCH_SIZE = 5
EPOCHS = 40
WARMPUP_STEPS_RATIO = 0.12

# 加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn)
print("success loading train_dataset...")

steps_per_epoch = len(train_dataset) // BATCH_SIZE  # 单次迭代步数
TOTAL_STEPS = steps_per_epoch * EPOCHS  # 总步数
WARMPUP_STEPS = TOTAL_STEPS * WARMPUP_STEPS_RATIO  # 学习率warm-up的步数

# 训练参数：全局步数、初始学习率、最终学习率
GLOBAL_STEPS = 1
LR_INIT = 5e-5
LR_END = 1e-6


# 学习率调度函数
def lr_func():
    if GLOBAL_STEPS < WARMPUP_STEPS:
        lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
        )
    return float(lr)


# 绘制损失折线图
def draw_train_loss(batch, all_train_loss):
    plt.title("Training Loss", fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(batch, all_train_loss, color='green', label='training loss')
    plt.ylim(0.5, 1.5)
    plt.legend()
    plt.grid()
    plt.savefig(f"../Results/train{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()
    plt.close()


print("start training...")
model.train()

epoch_loss = []
epoch_step_loss = []

for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.to(device)
        batch_boxes = batch_boxes.to(device)
        batch_classes = batch_classes.to(device)

        lr = lr_func()
        for param in optimizer.param_groups:
            param['lr'] = lr

        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.backward()
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)

        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e" % \
              (
                  GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0], losses[1], losses[2], cost_time,
                  lr))
        epoch_step_loss.append(loss.item())
        GLOBAL_STEPS += 1

    epoch_loss.append(loss.item())
    torch.save(model.state_dict(), "./weights/aircraft_512x800_epoch%d_loss%.4f.pth" % (epoch + 1, loss.item()))

# 输出损失折线图
try:
    # 全局损失折线图
    batch = [i + 1 for i in range(12000)]
    draw_train_loss(batch, epoch_step_loss)
    # 迭代阶段损失折线图
    batch = [i + 1 for i in range(40)]
    draw_train_loss(batch, epoch_loss)
except Exception as e:
    print(str(e))
