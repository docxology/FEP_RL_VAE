"""Plotting utilities for FEP-RL-VAE training visualization."""

import os
import math
import matplotlib.pyplot as plt


def plot_images(images, title, show=True, name="", folder=""):
    """Plot a grid of images."""
    n_images = len(images)
    if n_images == 0:
        # Handle empty list case
        fig = plt.figure(figsize=(4, 4))
        fig.suptitle(title)
        plt.text(0.5, 0.5, "No images to display", ha="center", va="center")
        if name:
            os.makedirs(f"images/{folder}", exist_ok=True)
            plt.savefig(f"images/{folder}/{name}.png")
        if show:
            plt.show()
        plt.close()
        return
    
    columns = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / columns)
    fig = plt.figure(figsize=(columns + 1, rows + 1))
    fig.suptitle(title)
    for i in range(1, rows * columns + 1):
        ax = fig.add_subplot(rows, columns, i)
        if i <= n_images:
            ax.imshow(images[i - 1], cmap="gray")
        ax.axis("off")
    if name:
        os.makedirs(f"images/{folder}", exist_ok=True)
        plt.savefig(f"images/{folder}/{name}.png")
    if show:
        plt.show()
    plt.close()


def plot_training_history(epoch_history):
    """Plot comprehensive training history."""

    plt.figure(figsize=(6, 6))
    for key, value in epoch_history["accuracy_losses"].items():
        plt.plot(value, label=f"accuracy loss {key}")
    for key, value in epoch_history["complexity_losses"].items():
        plt.plot(value, label=f"complexity loss {key}")
    plt.title("Losses for Accuracy and Complexity over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(epoch_history["total_reward"], label="total")
    plt.plot(epoch_history["reward"], label="reward")
    for key, value in epoch_history["curiosities"].items():
        plt.plot(value, label=f"curiosity {key}")
    plt.title("Rewards over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(epoch_history["actor_loss"], label="actor loss")
    for key, alpha_entropy in epoch_history["alpha_entropies"].items():
        plt.plot(alpha_entropy, label=f"alpha entropy {key}")
    for key, alpha_normal_entropy in epoch_history["alpha_normal_entropies"].items():
        plt.plot(alpha_normal_entropy, label=f"alpha normal entropy {key}")
    for key, total_entropy in epoch_history["total_entropies"].items():
        plt.plot(total_entropy, label=f"total entropy {key}")
    plt.title("Actor loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    for i, critic_loss in enumerate(epoch_history["critic_losses"]):
        plt.plot(critic_loss, label=f"critic {i} loss")
    plt.title("Critic loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()