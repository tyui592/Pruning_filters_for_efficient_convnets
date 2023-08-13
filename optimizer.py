import torch

def get_optimizer(network, args):
    # 神经网络优化器，主要是为了优化我们的神经网络，使他在我们的训练过程中快起来，节省社交网络训练的时间
    # SGD是最基础的优化方法，普通的训练方法, 需要重复不断的把整套数据放入神经网络NN中训练,
    # 这样消耗的计算资源会很大.当我们使用SGD会把数据拆分后再分批不断放入 NN 中计算.
    # 每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程, 而且也不会丢失太多准确率.
    # 实现梯度下降算法
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    # weight_decay
    # 为了有效限制模型中的自由参数数量以避免过度拟合，可以调整成本函数。
    # momentum
    # “冲量”这个概念源自于物理中的力学，表示力对时间的积累效应。
    # 在普通的梯度下降法x += v中，每次x的更新量v为v =−dx∗lr，其中dx为目标函数func(x)对x的一阶导数，。
    # 当使用冲量时，则把每次x的更新量v考虑为本次的梯度下降量−dx∗lr与上次x的更新量v乘上一个介于[0, 1][0, 1]的因子momentum的和，即
    # v =−dx∗lr + v∗momemtum

    scheduler = None
    if args.lr_milestone is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=args.lr_gamma)

    return optimizer, scheduler
