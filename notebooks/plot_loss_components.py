import matplotlib.pyplot as plt
import csv

epochs = []
total_loss = []
interior_loss = []
terminal_loss = []
boundary_loss = []
loss_boundary_mc = []

try:
    with open('training_log.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            epochs.append(int(row['epoch']))
            total_loss.append(float(row['total_loss']))
            interior_loss.append(float(row['loss_interior']))
            terminal_loss.append(float(row['loss_terminal']))
            boundary_loss.append(float(row['loss_boundary']))
            loss_boundary_mc.append(float(row['loss_boundary_mc']))

    # Plot the training loss components
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, total_loss, label='Total Loss')
    plt.plot(epochs, interior_loss, label='Interior Loss')
    plt.plot(epochs, terminal_loss, label='Terminal Loss')
    plt.plot(epochs, boundary_loss, label='Boundary Loss')
    plt.plot(epochs, loss_boundary_mc, label='loss_boundary_mc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss from training_log.csv')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("training_log.csv not found. Skipping loss plot.")
except Exception as e:
    print(f"An error occurred while plotting training_log.csv: {e}")
