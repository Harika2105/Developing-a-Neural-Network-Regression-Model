# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model

<img width="1079" height="704" alt="Screenshot 2026-02-02 203655" src="https://github.com/user-attachments/assets/5d0beb29-e0a8-44b9-9686-bdb4d741cc61" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Sivamani Harika

### Register Number: 212224240155

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')


```

### Dataset Information

<img width="467" height="295" alt="Screenshot 2026-02-02 204514" src="https://github.com/user-attachments/assets/2a88caee-baf3-4898-b106-8f7762ce2cc1" />


### OUTPUT

## Loss

<img width="552" height="227" alt="Screenshot 2026-02-02 205021" src="https://github.com/user-attachments/assets/9b13b200-547c-47a9-ac9e-2d8538b3cb1c" />

### Training Loss Vs Iteration Plot

<img width="802" height="584" alt="Screenshot 2026-02-02 205131" src="https://github.com/user-attachments/assets/6152543a-ca57-41dc-be57-42f43a4242c0" />


### New Sample Data Prediction

<img width="410" height="38" alt="Screenshot 2026-02-02 205159" src="https://github.com/user-attachments/assets/f92efdf4-71a9-48c4-9b1a-3fb64c44a34e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
