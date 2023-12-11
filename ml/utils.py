import pickle
import torch
from datetime import datetime
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
from dataset_loader.dataset import unnormalize_for_plot

def plot_sample(sample, model, device):
    """
    Plots a sample and its prediction by the model.

    Parameters:
    sample (torch.Tensor): The sample to be plotted.
    model (torch.nn.Module): The model used to generate the prediction.
    device (torch.device): The device where the computations will be performed.

    Returns:
    None
    """
    model.eval()
    with torch.no_grad():
        _, my_pred, mask = model(sample.to(device))
        unpatched_pred = model.mae.unpatchify(my_pred)

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.mae.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        # torch.Size([8, 4, 3, 224, 224])
        # [BATCH_SIZE x Frames x Channel x Height x Width ]

        for i in range(1):
            fig, axs = plt.subplots(1, 4)
            axs[0].imshow(unnormalize_for_plot(sample[i][0]))
            axs[1].imshow(unnormalize_for_plot(sample[i][1]))
            axs[2].imshow(1-mask[i])
            axs[3].imshow(unnormalize_for_plot(unpatched_pred[i]))  
            axs[0].set_title("Frame 1")
            axs[1].set_title("Frame 2")
            axs[2].set_title("Mask")
            axs[3].set_title("Predicted Frame 2")  
            plt.show()

def train_one_epoch(data_loader, optimizer, scheduler, device, model):
    """
    Trains the model for one epoch.

    Parameters:
    data_loader (torch.utils.data.DataLoader): The data loader.
    optimizer (torch.optim.Optimizer): The optimizer.
    scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
    device (torch.device): The device where the computations will be performed.
    model (torch.nn.Module): The model to be trained.

    Returns:
    losses (list): The list of loss values for each batch in the epoch.
    """
    losses = []
    for batch in tqdm(data_loader):
        batch = batch["video"].to(device)
        optimizer.zero_grad()
        loss = model(batch)[0]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        scheduler.step()
    return losses

def train(model, data_loader, epochs=60, prefix="", num_epochs_per_save=10, device=torch.device("cuda")):
    """
    Trains the model for a specified number of epochs.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    data_loader (torch.utils.data.DataLoader): The data loader.
    epochs (int): The number of epochs to train for.
    prefix (str): The prefix for the filenames of the saved losses and checkpoints.
    num_epochs_per_save (int): The number of epochs between each save of the model's state.
    device (torch.device): The device where the computations will be performed.

    Returns:
    None
    """
    # Get the first batch of video data from the data loader
    sample = next(iter(data_loader))["video"]

    # Get the current date and time, formatted as a string
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Define the path where the loss values will be stored
    file_to_store_losses_path = f"./losses/{prefix}{now}.txt" 

    # Define the directory where the model checkpoints will be stored
    folder_to_store_checkpoints = f"./checkpoints/{prefix}{now}"
    # Create the directory if it doesn't already exist
    os.makedirs(folder_to_store_checkpoints, exist_ok=True)

    # Initialize the optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    # Initialize the learning rate scheduler
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, eta_min=1e-6)

    # Loop over the specified number of epochs
    for epoch in range(epochs):
        # Print the current epoch number
        print(f"Epoch {epoch+1}")

        # Set the model to training mode
        model.train()
        # Train the model for one epoch and get the list of loss values
        losses = train_one_epoch(data_loader, optim, sched, device, model)
        # Step the scheduler
        sched.step()
        # Calculate the average loss for this epoch
        avg_loss = sum(losses)/len(losses)
        # Print the average loss
        print(f"Avg Loss: {avg_loss}")

        # Open the file to store the losses and append the average loss for this epoch
        with open(file_to_store_losses_path, "a") as f:
            f.write(f"Epoch {epoch+1} - Avg Loss: {avg_loss} Losses - {losses} \n")

        # Plot a sample and its prediction by the model
        plot_sample(sample, model, device)

        # If the current epoch number is a multiple of num_epochs_per_save, save the model's state
        if (epoch+1) % num_epochs_per_save == 0:
            torch.save(model, f"{folder_to_store_checkpoints}/{epoch+1}.pt")

def plot_results_from_checkpoint(checkpoint_name, data_loader, device, samples_path="samples.pickle"):
    """
    Loads a model from a checkpoint, evaluates it on a data loader, and plots the results for a set of samples.

    Parameters:
    checkpoint_name (str): The name of the checkpoint file.
    data_loader (torch.utils.data.DataLoader): The data loader.
    device (torch.device): The device where the computations will be performed.
    samples_path (str): The path to the pickle file containing the samples to be plotted.

    Returns:
    None
    """
    # Define the path to the checkpoint file
    checkpoint_path = f"./checkpoints/{checkpoint_name}"
    # Load the model from the checkpoint file and move it to the specified device
    model = torch.load(checkpoint_path).to(device)
    # Set the model to evaluation mode
    model.eval()

    # Initialize a list to store the loss values
    losses = []
    # Loop over the batches in the data loader
    for batch in tqdm(data_loader):
        # Move the batch to the specified device
        batch = batch["video"].to(device)
        # Compute the loss for this batch
        loss = model(batch)[0]
        # Append the loss value to the list
        losses.append(loss.item())
    # Compute the average loss
    avg_loss = sum(losses)/len(losses)
    # Print the average loss
    print(f"Avg Loss: {avg_loss}")

    # Create the directory for the test losses if it doesn't already exist
    os.makedirs("./test_losses", exist_ok=True)
    # Open the file to store the losses and append the average loss
    with open(f"./test_losses/{checkpoint_name.split('/')[0]}.txt", "a") as f:
        f.write(f"Avg Loss: {avg_loss} Losses - {losses} \n")

    # Open the pickle file containing the samples
    with open(samples_path, "rb") as f:
        # Load the samples
        samples = pickle.load(f)
        
    # Loop over the samples
    for sample in samples:
        # Plot the sample and its prediction by the model
        plot_sample(sample, model, device)
    # Delete the model to free up memory
    del model
