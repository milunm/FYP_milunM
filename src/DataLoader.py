from torch.utils.data import DataLoader #required pytorch class
from .dataset import PairedDataset  #import the PairedDataset class from dataset.py

# datasets

train_dataset = PairedDataset(split="train", size=256) #training dataset object
val_dataset = PairedDataset(split="val", size=256) #validation dataset object
test_dataset = PairedDataset(split="test", size=256) #test dataset object

# data loaders

train_loader = DataLoader(train_dataset, batch_size = 20, shuffle=True, pin_memory=True, num_workers=0) #training data loader with batch size of 20 and shuffling enabled
#shuffling ensures each epoch sees data in different order, improving generalization

val_loader = DataLoader(val_dataset, batch_size = 20, shuffle=False, pin_memory=True, num_workers=0) #validation data loader with batch size of 20 and no shuffling
#no shuffling as we want consistent evaluation

test_loader = DataLoader(test_dataset, batch_size = 20, shuffle=False, pin_memory=True, num_workers=0) #test data loader with batch size of 20 and no shuffling
#no shuffling as we want consistent evaluation in testing
# pin_memory=True speeds up data transfer to GPU

#parameters easy to tweak during training and to see diff results

x_batch, y_batch = next(iter(train_loader))
print(x_batch.shape, y_batch.shape)  #print the shapes of a single batch from the training loader to verify correct loading



