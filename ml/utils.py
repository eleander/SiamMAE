import torch
from datetime import datetime
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
from dataset_loader.dataset import unnormalize_for_plot

def plot_sample(sample, model, device):
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

def train_one_epoch(data_loader, optimizer,scheduler, device, model):
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
    sample = next(iter(data_loader))["video"]

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    file_to_store_losses_path = f"./losses/{prefix}{now}.txt" 

    folder_to_store_checkpoints = f"./checkpoints/{prefix}{now}"
    os.makedirs(folder_to_store_checkpoints, exist_ok=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, eta_min=1e-6)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        model.train()
        losses = train_one_epoch(data_loader, optim, sched, device, model)
        sched.step()
        avg_loss = sum(losses)/len(losses)
        print(f"Avg Loss: {avg_loss}")

        with open(file_to_store_losses_path, "a") as f:
            f.write(f"Epoch {epoch+1} - Avg Loss: {avg_loss} Losses - {losses} \n")

        plot_sample(sample, model, device)

        # save checkpoint of model after each epoch in folder_to_store_checkpoints/{epoch_number}.pt
        if (epoch+1) % num_epochs_per_save == 0:
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            torch.save(model, f"{folder_to_store_checkpoints}/{epoch+1}.pt")

def plot_results_from_checkpoint(checkpoint_name, data_loader, device):
    checkpoint_path = f"./checkpoints/{checkpoint_name}"
    model = torch.load(checkpoint_path).to(device)
    sample = next(iter(data_loader))["video"]
    plot_sample(sample, model, device)
    del model
