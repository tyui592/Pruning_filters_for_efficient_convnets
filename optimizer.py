import torch

def get_optimizer(network, args):
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)    

    scheduler = None
    if args.lr_milestone is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=args.lr_gamma)

    return optimizer, scheduler
