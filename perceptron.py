#! usr/bin/python

# Make predictions on a dataset based on current weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# Estimate perceptron weights usign Stochastic Gradient Descent
def train_weights(train_set, lear_rate, n_epoch):
    weights = [0.0 for i in range(len(train_set[0]))]
    for epoch in range(n_epoch):
        sqrd_err_sum = 0.0
        for row in train_set:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sqrd_err_sum += error**2
            # Adjusting the bias term
            weights[0] = weights[0] + lear_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + lear_rate * error * row[i]

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lear_rate, sqrd_err_sum))
    return weights

# Training dataset
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

lear_rate = 0.1
n_epoch = 5
weights = train_weights(dataset, lear_rate, n_epoch)
print(weights)
