#%%
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch

class QCDataLoader:

    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ids, self.masks = self.dataset.process_texts()
        self.create_loaders()
    
    def create_loaders(self):
        """
        Create Torch dataloaders for data splits
        """
        inputs = torch.tensor(self.ids)
        masks = torch.tensor(self.masks)
        
        data = TensorDataset(inputs, masks)
        sampler = SequentialSampler(data)
        self.dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

# %%

class FGGDataLoader:
    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.X = self.dataset.preprocess()
        self.create_loaders()
    
    def create_loaders(self):
        """
        Create Torch dataloaders for data splits
        """
        # Make sure everything is ok
        assert self.X['left'].shape == self.X['right'].shape
        left = torch.tensor(self.X['left'])
        right = torch.tensor(self.X['right'])

        data = TensorDataset(left, right)
        sampler = SequentialSampler(data)
        self.dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
# %%
class FGDataLoader:
    ''' This class is written considering that input is a 
        list of lists where each list is contains the similar cluster of sentences '''

    '''Not sure how to deal with batches here. '''

    def __init__(self,dataset):
        self.dataset = dataset
        self.inputs = self.dataset.preprocess()
        # self.create_loader()

    def create_loader(self):
        # TODO implement batching
        pass
    #    sampler = SequentialSampler(self.inputs)
    #    self.dataloader = DataLoader(self.inputs, sampler=sampler)

#%%
