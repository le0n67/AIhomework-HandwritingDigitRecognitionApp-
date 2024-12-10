# 人工智能课程作业：手写数字识别

## 1. 安装库

请确保您的环境中已经安装了Python和pip。然后运行以下命令来安装所需的库。

### 基本库
```shell
pip install torch torchvision numpy Pillow matplotlib
```

**若下载速度过慢，请配置国内pip镜像**

```shell
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 2. 运行程序

### 下载数据集并初始化模型

首次运行程序时会自动下载MNIST数据集（大约需要2-3分钟），请保持网络畅通。

```shell
python handwriting_recognition.py
```

**说明：**

- 首次运行时，程序会训练一个简单的神经网络模型并保存到`mnist_net.pth`文件中。
  <img src="https://xintakeout.oss-cn-beijing.aliyuncs.com/blog/202412101059377.png" alt="image-20241210105915328" style="zoom:50%;" />
- 之后的运行会加载预训练的模型，直接进入手写数字识别界面。<img src="https://xintakeout.oss-cn-beijing.aliyuncs.com/blog/202412101059788.png" alt="image-20241210105933763" style="zoom:80%;" />

### 使用界面

1. **启动应用**
   
   ```shell
   python handwriting_recognition.py
   ```
   
2. **操作步骤**
   - 在画板上使用鼠标绘制数字。
   - 点击“识别”按钮，程序将预测所绘制的数字并在控制台打印结果。
   - 点击“清除”按钮，清空画板以便重新绘制。
     <img src="https://xintakeout.oss-cn-beijing.aliyuncs.com/blog/202412101100894.png" alt="image-20241210110035860" style="zoom:67%;" />

## 3. 代码来源

本项目基于[@kongfanhe](https://gitee.com/kongfanhe)在[Gitee](https://gitee.com/kongfanhe/pytorch-tutorial.git)上的`test.py`文件进行修改，增加了Tkinter图形用户界面（GUI）功能。