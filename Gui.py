import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from Models.MLP.Layer import Layer
from Models.MLP.Network import Mlp
from Data import *
from utilities import *

def run_test_gui():
    # Retrieve inputs
    try:
        hidden_layers = int(hidden_layers_entry.get())
        neurons_per_layer = list(map(int, neurons_entry.get().split(",")))
        learning_rate = float(learning_rate_entry.get())
        epochs = int(epochs_entry.get())
        add_bias = bias_var.get()
        activation_function = activation_var.get()

        # Validate inputs
        if len(neurons_per_layer) != hidden_layers:
            raise ValueError("Number of neurons must match the number of hidden layers.")

        # Generate data and create MLP
        data = Data()
        train_input, train_output, test_input, test_output = data.GenerateData(['A', 'B', 'C'])
        mlp = Mlp(learning_rate=learning_rate, epochs=epochs)

        # Create layers
        layers = []
        input_layer = Layer(False, neurons_per_layer[0])  # Input layer
        input_layer.init_with_size(5)  # Number of features = 5
        layers.append(input_layer)

        for i in range(hidden_layers):
            hidden_layer = Layer(add_bias, 3 if i + 1 == hidden_layers else neurons_per_layer[i+1])  # Add bias if selected
            hidden_layer.init_with_size(neurons_per_layer[i])
            layers.append(hidden_layer)

        output_layer = Layer(True, 0)  # Output layer
        output_layer.init_with_size(3)  # Number of classes = 3
        layers.append(output_layer)

        mlp.training_layers = layers

        # Fit the model
        acc = mlp.fit(train_input, train_output)
        print(acc)
        # Calculate accuracy
        accuracy = 0
        for i, j in zip(test_input, test_output):
            accuracy += int(mlp.get_output(i) == j)
        accuracy /= len(test_input)

        confusion_matrix(test_output, [mlp.get_output(x) for x in test_input])

        # Display results
        messagebox.showinfo("Results", f"Accuracy: {accuracy:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create the Tkinter GUI
root = tk.Tk()
root.title("MLP Configuration")

# User Input Fields
tk.Label(root, text="Number of Hidden Layers:").grid(row=0, column=0, sticky="w")
hidden_layers_entry = tk.Entry(root)
hidden_layers_entry.grid(row=0, column=1)

tk.Label(root, text="Neurons per Layer (comma-separated):").grid(row=1, column=0, sticky="w")
neurons_entry = tk.Entry(root)
neurons_entry.grid(row=1, column=1)

tk.Label(root, text="Learning Rate (eta):").grid(row=2, column=0, sticky="w")
learning_rate_entry = tk.Entry(root)
learning_rate_entry.grid(row=2, column=1)

tk.Label(root, text="Number of Epochs (m):").grid(row=3, column=0, sticky="w")
epochs_entry = tk.Entry(root)
epochs_entry.grid(row=3, column=1)

bias_var = tk.BooleanVar()
tk.Checkbutton(root, text="Add Bias", variable=bias_var).grid(row=4, columnspan=2, sticky="w")

tk.Label(root, text="Activation Function:").grid(row=5, column=0, sticky="w")
activation_var = tk.StringVar(value="Sigmoid")
activation_menu = ttk.Combobox(root, textvariable=activation_var, values=["Sigmoid", "Tanh"])
activation_menu.grid(row=5, column=1)

# Run Button
run_button = tk.Button(root, text="Run", command=run_test_gui)
run_button.grid(row=6, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
