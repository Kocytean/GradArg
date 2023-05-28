import torch
from af import *

model = ExampleLRModel()
model.load_state_dict(torch.load('model.pt'))

data = SampleIterator()
print(f'Found {len(data)} data samples')

loss = torch.nn.MSELoss()
af = AF(model, data, loss)
print([(g.i, g.rank, g.clean) for g in af.grads])