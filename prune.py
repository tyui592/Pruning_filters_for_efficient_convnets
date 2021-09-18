import torch

from network import VGG
from train import train_network

def prune_network(args, network=None):
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")

    if network is None:
        network = VGG(args.vgg, args.data_set)
        # 加载网络
        if args.load_path:
            check_point = torch.load(args.load_path)
            network.load_state_dict(check_point['state_dict'])

    # prune network 输入需要剪的层号和通道数量
    network = prune_step(network, args.prune_layers, args.prune_channels, args.independent_prune_flag)
    network = network.to(device)
    print("-*-"*10 + "\n\tPrune network\n" + "-*-"*10)
    print(network)

    if args.retrain_flag:
        # update arguemtns for retraing pruned network
        args.epoch = args.retrain_epoch
        args.lr = args.retrain_lr
        args.lr_milestone = None # don't decay learning rate

        network = train_network(args, network)

    return network

def prune_step(network, prune_layers, prune_channels, independent_prune_flag):
    '''
        prune_layers：需要剪的层号
        prune_channels：需要剪掉的通道数量
        independent_prune_flag：剪枝方法是独立的还是贪婪的
        （作者给的原输入非常简单只剪了前两层各一个）
    '''
    network = network.cpu()# 剪枝主要是cpu上进行操作

    count = 0 # count for indexing 'prune_channels'
    conv_count = 1 # conv count for 'indexing_prune_layers'
    dim = 0 # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    # 如果是0.表示输入不变，将这个卷积核的输出给去掉，同时一连串后面的bn，以及后面对应的对应的卷积核也需要剪掉，
    # 如果是1，表示要把前面的featuremaps给剪掉。
    residue = None # residue is need to prune by 'independent strategy' 残差
    for i in range(len(network.features)):
        # 循环所有层，提取卷积层
        if isinstance(network.features[i], torch.nn.Conv2d):
            if dim == 1:
                # 当前是1，表明上一层的filters被剪了，所以这一层要将inchannel的filters按照channel_index同时给剪掉
                # 不管哪个层，上一层被剪了，就需要修改输入通道数
                new_, residue = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1
                # 当前是0，表明我们这一层要把输出的filters给剪掉。同时得到channel_index

            if 'conv%d'%conv_count in prune_layers:
            # 读取需要被剪枝的卷积层层号
                # 排序输出需要剪掉的fliter索引
                channel_index = get_channel_index(network.features[i].weight.data, prune_channels[count], residue)
                # 获取新的卷积层参数
                new_ = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1 # ^=异或运算
                count += 1
            else:
                residue = None
            conv_count += 1
        # bn层也是有通道的，需要将bn层同样做下处理。
        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(network.features[i], channel_index)
            network.features[i] = new_

    # update to check last conv layer pruned
    if 'conv13' in prune_layers:
        network.classifier[0] = get_new_linear(network.classifier[0], channel_index)

    return network

def get_channel_index(kernel, num_elimination, residue=None):
    '''
        获取用于修剪的候选通道索引
    '''
    # get cadidate channel index for pruning
    ## 'residue' is needed for pruning by 'independent strategy'
    # 绝对值排序，按照最小值挑出前num_elimination个的下标
    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    
    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()

def index_remove(tensor, dim, index, removed=False):
    # 根据index进行剪枝
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))
    
    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor

def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data

        return new_conv, residue

def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)
        
    return new_norm

def get_new_linear(linear, channel_index):
    # 全连接因为filters数目的变化，也需要进行变化
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                out_features=linear.out_features,
                                bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data
    
    return new_linear
