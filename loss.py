import torch

class Loss_Calculator(object):
    def __init__(self):
        # 交叉熵损失函数
        self.criterion = torch.nn.CrossEntropyLoss()        
        self.loss_seq = []
    
    def calc_loss(self, output, target):
        '''
        计算loss
        '''
        loss = self.criterion(output, target)        
        self.loss_seq.append(loss.item())
        return loss

    def get_loss_log(self, length=100):
        '''
        获得最近一百个loss的平均损失值
        '''
        # get recent average loss values
        if len(self.loss_seq) < length:
            length = len(self.loss_seq)
        return sum(self.loss_seq[-length:])/length
