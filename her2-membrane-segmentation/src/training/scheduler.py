from torch.optim.lr_scheduler import CosineAnnealingLR

def get_scheduler(optimizer, config):
    return CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
