from dataloaders.datasets import dataset
from torch.utils.data import DataLoader

def make_data_loader(args):
    train_set = dataset.Segmentation(args, split='train')
    val_set = dataset.Segmentation(args, split='val')
    test_set = dataset.Segmentation(args, split='test')
    
    train_loader = DataLoader(train_set, batch_size=args.tr_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader