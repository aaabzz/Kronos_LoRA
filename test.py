# coding:utf-8

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


# 使用单个Parquet执行一次完整训练冒烟测试。
# 改造Dataset分片。
# 最后接入双卡DDP。