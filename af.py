import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import numpy as np
import scipy

def pretty_print_dictionary(d, indent=0):
	for key, value in d.items():
		print('\t' * indent + str(key) + '\t' + str(value))

class SampleIterator(Dataset):
	def __init__(self, m= -5, c = 0, steps = 8, noise_factor = 1):
		super(SampleIterator, self).__init__()
		'''Sample data iterator for training a linear model'''
		self.x = torch.arange(-2, 2.01, 4/steps).view(-1, 1)
		self.y = (m*self.x) + (noise_factor * torch.randn(self.x.size())) + c
		self.l = len(self.x)
	def __len__(self):
		return self.l
	def __getitem__(self,index):
		return self.x[index], self.y[index]

class ExampleLRModel(torch.nn.Module):
	def __init__(self):
		super(ExampleLRModel, self).__init__()
		self.w = torch.nn.Parameter(torch.tensor(-10.0))
		self.b = torch.nn.Parameter(torch.tensor(0.0))
	def forward(self,x):
		return self.w * x + self.b

class GradVector:
	def __init__(self, i, grad, rank = 0, clean = True):
		self.i = i
		self.grad = np.array(grad)
		self.rank = None
		self.clean = clean
	def __add__(self, adduct):
		return self.grad + adduct
	def __radd__(self, adduct):
		return self.__add__(adduct)
	def __iadd__(self, adduct):
		self.grad = self.__add__(adduct)
		return self
	def __sub__(self, adduct):
		return self.grad - adduct
	def __rsub__(self, adduct):
		return adduct - self.grad
	def __isub__(self, adduct):
		self.grad = self.__sub__(adduct)
		return self

class AF:
	'''Builder for Quantitative Bipolar Argumentation Framework for agreement between sample datapoints over a trained model and given loss function'''
	def __init__(self, model, iterator, loss, forced_error=1, error = 5):

		self.model = model
		self.iterator = iterator
		self.num_args = len(iterator)
		self.loss = loss
		self.grads = []
		self.forced_error = forced_error
		self.error = error
		i = 0
		for x, y in self.iterator:
			self.model.zero_grad()
			self.loss(self.model(x), y+self.forced_error if i==(error-1) else y).backward()
			self.grads.append(GradVector(i, np.array([param.grad.view(-1).item() for param in model.parameters()]), 0, i!=(error-1)))
			i+=1
		self.model.zero_grad()
		maxgrad = max([np.linalg.norm(g.grad) for g in self.grads])
		for g in self.grads:
			g.grad = g.grad/maxgrad
		self.meangrad = np.mean([g.grad for g in self.grads], axis = 0)
		self.ranker = Ranker(self.grads, self.meangrad)
		self.ranks = self.ranker.eval()

class Ranker:
	'''Builder for Quantitative Bipolar Argumentation Framework for agreement between sample datapoints over a trained model and given loss function'''
	def __init__(self, grads, meangrad, rank = 0, iterations = 1):
		self.iterations = iterations
		self.grads = grads
		self.num_args = len(grads)
		self.strengths = np.array([1.0 for _ in range(self.num_args)])
		self.rank = rank
		self.weights = {} #
		i = 0
		self.meangrad = meangrad
		meannorm = np.linalg.norm(self.meangrad)
		for i in range(self.num_args):
			self.strengths[i] = meannorm/max(np.linalg.norm(self.grads[i].grad-self.meangrad),0.01)
			self.grads[i].rank = self.rank
			for j in range(i):
				self.weights[(j,i)]=0.001*np.linalg.norm(self.grads[i].grad+self.grads[j].grad)/(2*max(np.linalg.norm(self.grads[i].grad-self.grads[j].grad),0.01))
		self.base_score = scipy.special.softmax(self.strengths)

	def eval(self):
		if self.num_args>1:
			for _ in  range(self.iterations):
				aggregates = np.zeros(self.num_args)
				for i in range(1,self.num_args):
					for j in range(i):
						aggregates[j]+=self.weights[(j,i)]*self.strengths[i]
						aggregates[i]+=self.weights[(j,i)]*self.strengths[j]
				self.strengths = scipy.special.softmax(np.array([np.log(self.base_score[i]/(1-self.base_score[i])) + aggregates[i] for i in range(self.num_args)]))
			grad_x_strength = sorted([(self.grads[i], self.strengths[i]) for i in range(self.num_args)], key = lambda x: -x[1])
			top, bottom = grad_x_strength[:int(self.num_args/2)], grad_x_strength[int(self.num_args/2):]
			return [g[0] for g in top] + Ranker([g[0] for g in bottom], self.meangrad, self.rank+1, self.iterations).eval()
		else:
			return self.grads

def risk_error(model, data, loss, forced_error):
	grads = AF(model, data, loss, forced_error).grads
	eg = [g for g in grads if not g.clean][0].rank
	max_risk = max(g.rank for g in grads)
	return (max_risk-eg)/max_risk

def train_and_test(training_data, eval_data, forced_error = 2, loss=None, epochs = 10, lr = 0.00001):
	if loss is None:
		loss = torch.nn.MSELoss()
	errors = []
	losses = []
	model = ExampleLRModel()
	errors.append(risk_error(model, eval_data, loss, forced_error))
	optimiser = torch.optim.SGD(model.parameters(), lr = lr)
	for _ in range(epochs):
		epoch_loss = []
		for x, y in training_data:
			optimiser.zero_grad()
			l = loss(model(x),y)
			epoch_loss.append(l.item())
			l.backward()
			optimiser.step()
		losses.append(np.mean(epoch_loss))
		errors.append(risk_error(model, eval_data, loss, forced_error))
	del model
	return errors, losses