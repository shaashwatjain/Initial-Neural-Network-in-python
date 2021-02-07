'''
Design of a Neural Network from scratch

*************<IMP>*************
The hyperparameters tuned are-
	- np.random.seed(3) - We have used 3 as the seed value as it gave us optimal random values.
	- sigmoid - the activation function we have used for our implementation is the sigmoid function.
	- mse_loss - This is our loss function. It takes two np arrays and gives us the loss value using Mean Squared Error.
	- layers - We have taken 3 layers for our implementation -
															1) input layer with 9 inputs.
															2) hidden layer with 5 neurons.
															3) output layer with 1 neuron.
	- learning rate - The initial learning rate for our implementation is 0.05. We increase it by 0.01
	  after training our model with one batch of the dataset. Increasing the learning rate prevented 
	  our model from getting stuck in local minima and also helped in training our model better.
	- epochs- The iterations of one batch. The number of epochs is tuned at 200 for our model.
	- 0.6 is our deciding factor to determine as to which label it will belong to i.e 1 or 0.
	  If output is greater than or equal to 0.6 then we assign it a value of 1 otherwise 0.
	- Our dataset is split into an 80-20 train test data. ue to the small size of the dataset
	  we decided to go with 80-20 instead of 70-30
	- Our training data is further split into random batches of 70-30 to randomize input, therfore 
	  increasing efficiency and accuracy.
'''
# importing modules
import numpy as np		# for mathematical calculations and handling array operations 
import pandas as pd 	# for reading and handling data
from sklearn.model_selection import train_test_split	# to split dataset into test set and training set

np.seterr(all='ignore')
np.random.seed(3)

# sigmoid activation function
def sigmoid(x):
	x = np.float32(x)
	return 1/(1+ np.exp(-x))

# differential of sigmoid function 
def sigmoidDerivative(x):
	x = np.float32(x)
	o = sigmoid(x)
	return o * (1 - o)

# MSE loss function 
def mse_loss(truey, predy):
  return (((truey - predy) ** 2).mean())

# typecasting dataset
''' This function takes the dataset, converts each column into a list and the for loop organises it in the proper 
	form of input and output.
	This function makes our dataset ready for our model implementation.
'''
def typecastData(df):
	community = df['Community'].tolist()
	age = df['Age'].tolist()
	weight = df['Weight'].tolist()
	delphase = df['Delivery phase'].tolist()
	hb = df['HB'].tolist()
	ifa = df['IFA'].tolist()
	bp = df['BP'].tolist()
	education = df['Education'].tolist()
	residence = df['Residence'].tolist()
	y = df['Result'].tolist()

	X = []
	for i in range(df.shape[0]):
			X.append([community[i], age[i], weight[i], delphase[i],
								hb[i], ifa[i], bp[i], education[i], residence[i]])
	return X, y

# Class neural network
class NN:
	def __init__(self, layers=[9, 5, 1]):		#Optional implementation to change layers and neurons. DO NOT CHANGE the number of layers as that part is hardcoded.
												#You can change the hidden layers and see the difference in output.
		self.weights = []
		self.biases = []
		self.layers = layers
		x = 0
		for i in self.layers[1:]:
			for k in range(i):
				tlist = []
				for j in range(self.layers[x]):
					tlist.append(np.random.randn())		# Initializing weights to random values set by our seed value
				self.weights.append(tlist)
			x += 1
		#print(self.weights)

		myiter = sum(self.layers[1:])
		for i in range(myiter):
			self.biases.append(np.random.random())		#Initialising bias to random values set by our seed value
		#print(self.biases)

	def forwardProp(self, x):
		'''
		This function calculates weight*input + bias value for every neuron and passes it through the 
		activation	function to get the final output with current values
		'''
		sumLayer = []
		a = 0
		for i in range(self.layers[1]):
			tlist = []
			for j in range(len(self.weights[a])):
				tlist.append(self.weights[a][j]*x[j])
			tlist.append(self.biases[a])
			sumLayer.append(sigmoid(sum(tlist)))	#sumLayer stores the value which is to be propogated to the next layer
			a += 1
		tlist2 = []
		for i in range(len(self.weights[a])):
			tlist2.append(self.weights[a][i]*sumLayer[i])	#tlist2 combines the output of neuron with weights to pass to the next layer
		tlist2.append(self.biases[a])
		o = sigmoid(sum(tlist2))
		return o

	def fit(self,X,Y,lr = 0.05):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input.
		Default learning rate for the model is set to 0.05. Can be changed.
		Epochs is set to 200 here.
		'''
		epochs = 200
		for epoch in range(epochs):
			for x,truey in zip(X,Y):
				sumLayer = []
				# FINDING THE FEEDFORWARD OUTPUT AND SUM OF EVERY LAYER
				a = 0
				for i in range(self.layers[1]):
					tlist = []
					for j in range(len(self.weights[a])):
						tlist.append(self.weights[a][j]*x[j])
					tlist.append(self.biases[a])
					sumLayer.append(sigmoid(sum(tlist)))	#tlist contains the values weight*input + biases for layer 1
					a += 1
				tlist2 = []
				for i in range(len(self.weights[a])):
					tlist2.append(self.weights[a][i]*sumLayer[i])
				tlist2.append(self.biases[a])
				oSum = sum(tlist2)
				sumLayer.append(oSum)
				predy = sigmoid(oSum)

				der_predy = -2 * (truey - predy)	#der_predy holds the derived error

				#Calculating the derivative of any value wrt the bias
				der_any_wrt_b = []
				for i in range(len(self.biases)):
					der_any_wrt_b.append(sigmoidDerivative(sumLayer[i]))
				
				#Calculating the derivative of hidden layer input wrt weights
				der_h_wrt_w = []
				for i in range(self.layers[1]):
					temp = []
					for j in range(len(self.weights[i])):
						temp.append(x[j]*sigmoidDerivative(sumLayer[i]))
					der_h_wrt_w.append(temp)
				
				#Calculating the derivative of predicted value wrt hidden layer
				der_predy_wrt_h = []
				for i in range(self.layers[1]):
					der_predy_wrt_h.append(self.weights[-1][i]*sigmoidDerivative(oSum))

				#Calculating the derivative of predicted value wrt certain weights
				der_predy_wrt_w = []
				for i in range(self.layers[1]):
					der_predy_wrt_w.append(sumLayer[i]*sigmoidDerivative(oSum))
				
				# chain differentiation and multiplication to alter further value. Back Propogation for hidden layer.
				for i in range(self.layers[1]):
					for j in range(len(self.weights[i])):
						self.weights[i][j] -= lr * der_predy * der_predy_wrt_h[i] * der_h_wrt_w[i][j]
					self.biases[i] -= lr * der_predy * der_predy_wrt_h[i] * der_any_wrt_b[i]

				# updating weights and biases of output layer
				for i in range(self.layers[1]):
					self.weights[-1][i] -= lr * der_predy * der_predy_wrt_w[i]
				self.biases[-1] -= lr * der_predy * der_any_wrt_b[-1]

	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		# simple forward propagation 
		yhat = []
		for x in X:
			yhat.append(self.forwardProp(x))
		return yhat


	def CM(self, y_test, y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		# applying formulae 
		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		a = (tp+tn)/(tp+tn+fp+fn)

		# displaying out 
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print(f"Accuracy: {a*100}%")

# ----------------MAIN CODE----------------

# reading data 
data = pd.read_csv('../data/cleaned_Dataset.csv')
df = pd.DataFrame(data)

X, y = typecastData(df)

# splitting dataset into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model creation 
model = NN()

lr = 0.05
# training with batches of training data (BATCH PROCESSING)
for i in range(2):
	a,b,c,d = train_test_split(X_train,y_train,test_size = 0.7)
	model.fit(a,c,lr)
	lr = lr + 0.01

# prediction 
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# confusion matrix printing for final accuracy
print('*'*50)
print("\nTraining Accuracy")
model.CM(y_train, y_pred_train)
print('*'*50)
print("\nTesting Accuracy")
model.CM(y_test, y_pred_test)