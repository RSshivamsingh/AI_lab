import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_csv_data(filename_x, filename_y):

    X = pd.read_csv(filename_x).values.reshape(-1, 1) 
    y = pd.read_csv(filename_y).values.flatten() 
    return X, y

def hypothesis(theta_0, theta_1, X):

    return theta_0 + theta_1 * X

def cost_function(theta_0, theta_1, X, y):
    m = len(X)
    error_sum = 0.0
    for i in range(m):
        error_sum += (hypothesis(theta_0, theta_1, X[i]) - y[i]) ** 2
    return error_sum / (2 * m)

def gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-5):

    theta_0 = 0.0
    theta_1 = 0.0
    m = len(X)
    cost_history = [] 

    for i in range(max_iterations):
        gradient_theta_0 = 0.0
        gradient_theta_1 = 0.0

        for j in range(m):
            error = hypothesis(theta_0, theta_1, X[j]) - y[j]
            gradient_theta_0 += error
            gradient_theta_1 += error * X[j]

        theta_0 -= learning_rate * (gradient_theta_0 / m)
        theta_1 -= learning_rate * (gradient_theta_1 / m)

        current_cost = cost_function(theta_0, theta_1, X, y)
        cost_history.append(current_cost)

        if i > 0 and abs(current_cost - cost_history[-2]) < tolerance:
            break

    return theta_0, theta_1, current_cost, cost_history

def stochastic_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-5):

    theta_0 = 0.0
    theta_1 = 0.0
    m = len(X)
    cost_history = []

    for i in range(max_iterations):
        for j in range(m):
            random_index = np.random.randint(m)
            X_j = X[random_index]
            y_j = y[random_index]
            error = hypothesis(theta_0, theta_1, X_j) - y_j
            theta_0 -= learning_rate * error
            theta_1 -= learning_rate * error * X_j

            current_cost = cost_function(theta_0, theta_1, X, y)
            cost_history.append(current_cost)

            if i > 0 and abs(current_cost - cost_history[-2]) < tolerance:
                break

    return theta_0, theta_1, current_cost, cost_history

def mini_batch_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-5, batch_size=32):

    theta_0 = 0.0
    theta_1 = 0.0
    m = len(X)
    cost_history = []

    for i in range(max_iterations):
        for j in range(0, m, batch_size):
            X_batch = X[j:j+batch_size]
            y_batch = y[j:j+batch_size]
            gradient_theta_0 = 0.0
            gradient_theta_1 = 0.0

            for k in range(len(X_batch)):
                error = hypothesis(theta_0, theta_1, X_batch[k]) - y_batch[k]
                gradient_theta_0 += error
                gradient_theta_1 += error * X_batch[k]

            theta_0 -= learning_rate * (gradient_theta_0 / len(X_batch))
            theta_1 -= learning_rate * (gradient_theta_1 / len(X_batch))

            current_cost = cost_function(theta_0, theta_1, X, y)
            cost_history.append(current_cost)

            if i > 0 and abs(current_cost - cost_history[-2]) < tolerance:
                break

    return theta_0, theta_1, current_cost, cost_history

if __name__ == "__main__":
    filename_x = 'C:\\Users\\KIIT\\Documents\\sem 6\\linearX.csv'
    filename_y = 'C:\\Users\\KIIT\\Documents\\sem 6\\linearY.csv'
    X, y = read_csv_data(filename_x, filename_y)

    # Question 1: Batch Gradient Descent with learning rate 0.5
    theta_0_batch, theta_1_batch, final_cost_batch, cost_history_batch = gradient_descent(X, y, learning_rate=0.5)
    print("Batch Gradient Descent:")
    print("Final theta_0:", theta_0_batch)
    print("Final theta_1:", theta_1_batch)
    print("Final Cost:", final_cost_batch)

    # Question 3: Plot cost function vs. iteration for first 50 iterations
    plt.figure()
    plt.plot(range(50), cost_history_batch[:50])
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function")
    plt.title("Batch Gradient Descent Cost baaaaaa")
    plt.show()

    # Question 4: Plot the data and the fitted line
    plt.figure()
    plt.scatter(X, y, label="Data")
    plt.plot(X, hypothesis(theta_0_batch, theta_1_batch, X), color='red', label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Data and Fitted Line")
    plt.show()

    # # Question 5: Test with different learning rates
    learning_rates = [0.05,0.5,5]
    for lr in learning_rates:
        theta_0, theta_1, final_cost, cost_history = gradient_descent(X, y, learning_rate=lr)
        plt.figure()
        plt.plot(range(50), cost_history[:50])
        plt.xlabel("Iterations")
        plt.ylabel("Cost Function")
        plt.title(f"Batch Gradient Descent Cost (lr={lr})")
        plt.show()

        print(f"Learning Rate: {lr}")
        print("Final theta_0:", theta_0)
        print("Final theta_1:", theta_1)
        print("Final Cost:", final_cost)
        print()

    # Question 6: Stochastic and Mini-batch Gradient Descent
    # Choose a suitable learning rate (e.g., 0.01)
    learning_rate = 0.05

    theta_0_stochastic, theta_1_stochastic, final_cost_stochastic, cost_history_stochastic = stochastic_gradient_descent(X, y, learning_rate=learning_rate)
    theta_0_minibatch, theta_1_minibatch, final_cost_minibatch, cost_history_minibatch = mini_batch_gradient_descent(X, y, learning_rate=learning_rate)

    plt.figure()
    plt.plot(range(len(cost_history_batch)), cost_history_batch, label="Batch")
    plt.plot(range(len(cost_history_stochastic)), cost_history_stochastic, label="Stochastic")
    plt.plot(range(len(cost_history_minibatch)), cost_history_minibatch, label="Mini-batch")
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function")
    plt.title("Comparison of Gradient Descent Methods")
    plt.show()