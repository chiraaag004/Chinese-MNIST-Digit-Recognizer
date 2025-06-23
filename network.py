import numpy as np
from tqdm import tqdm

def predict(network, input):
    # input = input.reshape(-1, 1)

    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, batch_size=32, verbose=True):
    samples = x_train.shape[0]

    if batch_size > samples:
        batch_size = samples

    for e in range(epochs):
        error = 0

        # Shuffle the dataset
        indices = np.random.permutation(samples)
        x_train = x_train[indices]
        y_train = y_train[indices]

        batch_bar = tqdm(range(0, samples, batch_size), desc=f"Epoch {e+1}", leave=False)
        for i in batch_bar:
        
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward
            output = x_batch
            for layer in network:
                output = layer.forward(output)

            # print(f"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
            # print(f"output shape: {output.shape}, sum(output[:,0]): {np.sum(output[:,0])}")
            # print(f"sample output[:,0]: {output[:,0]}")
            # print(f"sample y_batch[:,0]: {y_batch[:,0]}")

            # Compute loss
            batch_loss = np.mean(loss(y_batch, output))
            error += np.mean(loss(y_batch, output))
            batch_bar.set_postfix(loss=f"{batch_loss:.4f}")

            # Backward
            grad = loss_prime(y_batch, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= (samples // batch_size)
        if verbose and (e+1) % 10 == 0:
            print(f"{e + 1}/{epochs}, error={error:.6f}")
