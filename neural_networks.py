import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, activation='tanh'):
        np.random.seed(0)
        self.learning_rate = learning_rate  # Learning rate
        self.activation_function = activation  # Activation function

        # Xavier initialization for weights
        bound1 = np.sqrt(6 / (input_dim + hidden_dim))
        self.weights1 = np.random.uniform(-bound1, bound1, size=(input_dim, hidden_dim))
        self.bias1 = np.zeros((1, hidden_dim))
        bound2 = np.sqrt(6 / (hidden_dim + output_dim))
        self.weights2 = np.random.uniform(-bound2, bound2, size=(hidden_dim, output_dim))
        self.bias2 = np.zeros((1, output_dim))

        # Variables for storing activations and gradients
        self.pre_activation1 = None
        self.activation1 = None
        self.pre_activation2 = None
        self.output = None

        # Dictionary for visualization purposes
        self.gradients = {}

    def forward(self, inputs):
        # Hidden layer computations
        self.pre_activation1 = np.dot(inputs, self.weights1) + self.bias1
        if self.activation_function == 'tanh':
            self.activation1 = np.tanh(self.pre_activation1)
        elif self.activation_function == 'relu':
            self.activation1 = np.maximum(0, self.pre_activation1)
        elif self.activation_function == 'sigmoid':
            self.activation1 = 1 / (1 + np.exp(-self.pre_activation1))
        else:
            raise ValueError('Unknown activation function')

        # Output layer computations
        self.pre_activation2 = np.dot(self.activation1, self.weights2) + self.bias2
        self.output = 1 / (1 + np.exp(-self.pre_activation2))

        return self.output

    def backward(self, inputs, targets):
        num_samples = targets.shape[0]

        # Output layer error
        delta_output = (self.output - targets)

        # Gradients for weights and biases of output layer
        grad_weights2 = np.dot(self.activation1.T, delta_output) / num_samples
        grad_bias2 = np.sum(delta_output, axis=0, keepdims=True) / num_samples

        # Backpropagation to hidden layer
        if self.activation_function == 'tanh':
            delta_activation1 = np.dot(delta_output, self.weights2.T)
            delta_hidden = delta_activation1 * (1 - self.activation1 ** 2)
        elif self.activation_function == 'relu':
            delta_activation1 = np.dot(delta_output, self.weights2.T)
            delta_hidden = delta_activation1 * (self.pre_activation1 > 0)
        elif self.activation_function == 'sigmoid':
            delta_activation1 = np.dot(delta_output, self.weights2.T)
            sigmoid = 1 / (1 + np.exp(-self.pre_activation1))
            delta_hidden = delta_activation1 * sigmoid * (1 - sigmoid)
        else:
            raise ValueError('Unknown activation function')

        # Gradients for weights and biases of hidden layer
        grad_weights1 = np.dot(inputs.T, delta_hidden) / num_samples
        grad_bias1 = np.sum(delta_hidden, axis=0, keepdims=True) / num_samples

        # Update weights and biases
        self.weights1 -= self.learning_rate * grad_weights1
        self.bias1 -= self.learning_rate * grad_bias1
        self.weights2 -= self.learning_rate * grad_weights2
        self.bias2 -= self.learning_rate * grad_bias2

        # Store gradients for visualization
        self.gradients['grad_weights1'] = grad_weights1
        self.gradients['grad_bias1'] = grad_bias1
        self.gradients['grad_weights2'] = grad_weights2
        self.gradients['grad_bias2'] = grad_bias2

    def compute_loss(self, targets):
        # Compute cross-entropy loss
        num_samples = targets.shape[0]
        epsilon = 1e-8  # To prevent log(0)
        loss = -np.mean(targets * np.log(self.output + epsilon) + (1 - targets) * np.log(1 - self.output + epsilon))
        return loss

def generate_data(n_samples=200):
    np.random.seed(0)
    # Generate random data
    data = np.random.randn(n_samples, 2)
    # Assign labels based on a circular boundary
    labels = ((data[:, 0] ** 2 + data[:, 1] ** 2) > 1).astype(int)
    labels = labels.reshape(-1, 1)
    return data, labels

# Visualization update function
def update(frame_index, mlp, ax_input, ax_hidden, ax_gradient, data, labels):
    # Clear previous plots
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform multiple training steps
    for _ in range(10):
        mlp.forward(data)
        mlp.backward(data, labels)

    # Compute and display the loss
    current_loss = mlp.compute_loss(labels)
    print(f"Step {frame_index * 10}, Loss: {current_loss:.4f}")

    # Plot hidden layer activations
    hidden_outputs = mlp.activation1
    ax_hidden.scatter(
        hidden_outputs[:, 0],
        hidden_outputs[:, 1],
        hidden_outputs[:, 2],
        c=labels.ravel(),
        cmap='bwr',
        alpha=0.7
    )
    ax_hidden.set_title(f'Hidden Space at Step {frame_index * 10}')
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')

    # Plot decision boundary in hidden space
    weights2_flat = mlp.weights2.flatten()
    bias2_value = mlp.bias2.item()
    if weights2_flat[2] != 0:
        x_limits = ax_hidden.get_xlim()
        y_limits = ax_hidden.get_ylim()
        x_vals = np.linspace(x_limits[0], x_limits[1], 10)
        y_vals = np.linspace(y_limits[0], y_limits[1], 10)
        X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
        Z_vals = (-weights2_flat[0]*X_mesh - weights2_flat[1]*Y_mesh - bias2_value) / weights2_flat[2]
        ax_hidden.plot_surface(X_mesh, Y_mesh, Z_vals, alpha=0.3, color='green')

    # Plot decision boundary in input space
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Forward pass through the network
    pre_activation1 = np.dot(grid_points, mlp.weights1) + mlp.bias1
    if mlp.activation_function == 'tanh':
        activation1 = np.tanh(pre_activation1)
    elif mlp.activation_function == 'relu':
        activation1 = np.maximum(0, pre_activation1)
    elif mlp.activation_function == 'sigmoid':
        activation1 = 1 / (1 + np.exp(-pre_activation1))
    else:
        raise ValueError('Unknown activation function')
    pre_activation2 = np.dot(activation1, mlp.weights2) + mlp.bias2
    output = 1 / (1 + np.exp(-pre_activation2))
    Z = output.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=50, cmap='bwr', alpha=0.6)
    ax_input.scatter(data[:, 0], data[:, 1], c=labels.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title(f'Input Space at Step {frame_index * 10}')
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # Visualize network architecture and gradients
    layers = [2, 3, 1]
    positions_x = [0, 1, 2]
    positions_neurons = []

    for idx, layer_size in enumerate(layers):
        positions_y = np.linspace(0.1, 0.9, layer_size)
        positions_x_layer = np.full(layer_size, positions_x[idx])
        positions_neurons.append(list(zip(positions_x_layer, positions_y)))

    # Flatten neuron positions
    all_positions = [position for layer in positions_neurons for position in layer]

    # Draw neurons
    for position in all_positions:
        neuron_circle = Circle(position, 0.03, color='white', ec='black', zorder=4)
        ax_gradient.add_patch(neuron_circle)

    # Determine maximum gradient for scaling
    max_gradient = max(
        np.abs(mlp.gradients['grad_weights1']).max(),
        np.abs(mlp.gradients['grad_weights2']).max()
    )

    # Draw connections with gradients
    for i, (x_start, y_start) in enumerate(positions_neurons[0]):
        for j, (x_end, y_end) in enumerate(positions_neurons[1]):
            gradient_value = mlp.gradients['grad_weights1'][i, j]
            line_width = (np.abs(gradient_value) / max_gradient) * 5
            color = 'green' if gradient_value > 0 else 'red'
            ax_gradient.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=line_width)

    for i, (x_start, y_start) in enumerate(positions_neurons[1]):
        for j, (x_end, y_end) in enumerate(positions_neurons[2]):
            gradient_value = mlp.gradients['grad_weights2'][i, j]
            line_width = (np.abs(gradient_value) / max_gradient) * 5
            color = 'green' if gradient_value > 0 else 'red'
            ax_gradient.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=line_width)

    ax_gradient.axis('off')
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.set_title(f'Gradients at Step {frame_index * 10}')
    ax_gradient.set_aspect('equal')  # Ensure circles are not distorted

def visualize(activation, learning_rate, total_steps):
    data, labels = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, learning_rate=learning_rate, activation=activation)

    # Configure the plotting environment
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Adjust layout
    plt.tight_layout()

    # Create the animation
    ani = FuncAnimation(
        fig,
        partial(
            update,
            mlp=mlp,
            ax_input=ax_input,
            ax_hidden=ax_hidden,
            ax_gradient=ax_gradient,
            data=data,
            labels=labels
        ),
        frames=total_steps // 10,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    learning_rate = 0.1
    total_steps = 1000
    visualize(activation, learning_rate, total_steps)