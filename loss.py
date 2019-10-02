import torch

class Loss_Calculator(object):
    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()        
        self.loss_seq = []
    
    def calc_loss(self, output, target):
        loss = self.criterion(output, target)        
        self.loss_seq.append(loss.item())
        return loss

    def get_loss_log(self, length=100):
        # get recent average loss values
        if len(self.loss_seq) < length:
            length = len(self.loss_seq)
        return sum(self.loss_seq[-length:])/length
