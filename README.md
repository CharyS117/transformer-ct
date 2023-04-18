# 基于深度学习中的 Transformer 在 CT 成像中的探讨

本仓库包含了毕业论文《基于深度学习中的 Transformer 在 CT 成像中的探讨》的相关代码。

## 数据集

论文中使用的数据集来自 [Mayo 低剂量 CT 挑战赛](https://www.aapm.org/grandchallenge/lowdosect/)。默认情况下，程序会读取文件夹及子文件夹中所有配对的 `.IMA` 文件。如果需要使用不同的配对规则或数据集，可以通过在 `data/helper.py` 中继承 `Helper` 类并修改配置文件来实现。

## 示例

1. 下载 Mayo 数据集。
2. 将数据集分为训练集和验证集，分别放入 `dataset/train` 和 `dataset/val` 文件夹。
3. 运行以下命令开始训练：

```shell
python train.py config/demo.yml
```

训练结果将保存在 `log` 文件夹中。在验证集上表现最好的一代模型参数以及最后一代模型参数将默认保存在 `checkpoints` 文件夹。

## 自定义配置

您可以通过修改配置文件来自定义本模型和训练过程，详见 `config/demo.yml`。
此外，您还可以通过调整网络架构进行自定义，例如 `arch/ffn_k3.py` 中将前馈神经网络中的卷积核大小改为了 3。

## 预训练模型

|参数量|文件大小|下载地址|
|:----:|:----:|:----:|
|80M|650.1MB|[Dropbox](https://www.dropbox.com/s/g028y7sl4ny6a53/80m.pth?dl=0)|
|20M|161.3MB|[Dropbox](https://www.dropbox.com/s/ga4r9cq3db5wkkn/20m.pth?dl=0)|
|10M|90.9MB|[Dropbox](https://www.dropbox.com/s/lsbl0wjvebqytln/10m.pth?dl=0)|

有关测试预训练模型的详细信息，请参阅 `test.py`。