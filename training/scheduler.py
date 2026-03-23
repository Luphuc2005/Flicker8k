from torch.optim.lr_scheduler import StepLR

def get_scheduler(optimizer):
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    return scheduler