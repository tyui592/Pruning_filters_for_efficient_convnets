import time

import torch
import torchvision

from network import VGG
from loss import Loss_Calculator
from evaluate import accuracy
from utils import AverageMeter, get_data_set
from optimizer import get_optimizer

def train_network(args, network=None, data_set=None):
    # 将代码分配到设备
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")

    if network is None:
        # 根据输入看使用那个vgg模型和数据集
        network = VGG(args.vgg, args.data_set)
    network = network.to(device)

    if data_set is None:
        data_set = get_data_set(args, train_flag=True)
    # 定义loss类
    loss_calculator = Loss_Calculator()
    # 神经网络优化器，主要是为了优化我们的神经网络，冲量、正则、SGD（不需要每次全部读入数据，可以分批读入）等
    optimizer, scheduler = get_optimizer(network, args)
    # 恢复训练的标志
    if args.resume_flag:
        # 加载训练好的模型
        check_point = torch.load(args.load_path)
        # 将加载好的模型加载到网络中
        network.load_state_dict(check_point['state_dict'])
        loss_calculator.loss_seq = check_point['loss_seq']
        args.start_epoch = check_point['epoch'] # update start epoch
    # 开始训练
    print("-*-"*10 + "\n\tTrain network\n" + "-*-"*10)
    for epoch in range(args.start_epoch, args.epoch):
        # make shuffled data loader
        # 数据读取的重要接口，从数据库中每次抽出batch size个样本
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

        # train one epoch
        train_step(network, data_loader, loss_calculator, optimizer, device, epoch, args.print_freq)

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()

        torch.save({'epoch': epoch+1, 
                   'state_dict': network.state_dict(),
                   'loss_seq': loss_calculator.loss_seq},
                   args.save_path+"check_point.pth")
        
    return network

def train_step(network, data_loader, loss_calculator, optimizer, device, epoch, print_freq=100):
    network.train()
    # set benchmark flag to faster runtime
    # 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    torch.backends.cudnn.benchmark = True
    # 更新类
    data_time = AverageMeter()
    loss_time = AverageMeter()    
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 返回当前时间的时间戳
    tic = time.time()
    for iteration, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - tic)
        
        inputs, targets = inputs.to(device), targets.to(device)
        # 网络输出
        tic = time.time()
        outputs = network(inputs)
        forward_time.update(time.time() - tic)
        # 计算loss
        tic = time.time()
        loss = loss_calculator.calc_loss(outputs, targets)
        loss_time.update(time.time() - tic)
        
        tic = time.time()
        # 把梯度置零，将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 反向传播求梯度
        optimizer.step()
        backward_time.update(time.time() - tic)
        # Top-1，Top-5中的Top指的是一个图片中的概率前1和前5，不是所有图片中预测最好的1个或5个图片
        # 比如一共需要分10类，每次分类器的输出结果都是10个相加为1的概率值，
        # Top1就是这十个值中最大的那个概率值对应的分类恰好正确的频率，
        # 而Top5则是在十个概率值中从大到小排序出前五个，然后看看这前五个分类中是否存在那个正确分类，再计算频率。
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1,5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # 日志打印输出
        if iteration % print_freq == 0:
            logs_ = '%s: '%time.ctime()
            logs_ += 'Epoch [%d], '%epoch
            logs_ += 'Iteration [%d/%d/], '%(iteration, len(data_loader))
            logs_ += 'Data(s): %2.3f, Loss(s): %2.3f, '%(data_time.avg, loss_time.avg)
            logs_ += 'Forward(s): %2.3f, Backward(s): %2.3f, '%(forward_time.avg, backward_time.avg)
            logs_ += 'Top1: %2.3f, Top5: %2.4f, '%(top1.avg, top5.avg)
            logs_ += 'Loss: %2.3f'%loss_calculator.get_loss_log()
            print(logs_)            
                        
        tic = time.time()
    return None
