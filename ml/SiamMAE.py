import torch
import os 
import requests
from dataset_loader.dataset import get_kinetics_dataset, unnormalize_for_plot, unnormalize
from mae import models_mae


# "A cross decoder consists of decoder blocks with only cross-attention layers, where tokens from f2
# attend to the tokens from f1." From Siamese Masked Autoencoders" from SiamMAE paper
class CrossDecoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, f1, f2):
        # Reshape f1 and f2 to fit the input shape requirement of MultiheadAttention
        # which is (seq_len, batch_size, embed_dim)
        f1 = f1.transpose(0, 1)
        f2 = f2.transpose(0, 1)

        # MultiheadAttention takes in the query, key, value. Here we use f2 to attend to f1
        attn_output, _ = self.multihead_attn(f2, f1, f1)

        # Reshape the output back to the original shape
        attn_output = attn_output.transpose(0, 1)

        return attn_output
    

class SiamMAE(torch.nn.Module):
    """
    SiamMAE is a class that represents a Siamese Masked Autoencoder.
    It uses two consecutive frames from a video sequence and learns to predict the next frame.
    The first frame is passed through the MAE's encoder with no masking.
    The next frame is passed through the MAE's encoder with a specified mask ratio.
    The model then attempts to predict the missing pixels in the next frame.
    """

    def __init__(self, mae, mask_ratio = 0.75, device=torch.device("cuda")):
        """
        Initializes the SiamMAE class.

        Parameters:
        mae (torch.nn.Module): An instance of a Masked Autoencoder (MAE).
        mask_ratio (float): The ratio of masking in the input data.
        device (torch.device): The device where the computations will be performed.
        """
        super().__init__()
        self.mae = mae 
        # CrossDecoder is used to attend the tokens from the first frame to the tokens from the next frame.
        self.cross_decoder = CrossDecoder(embed_dim=mae.decoder_embed_dimensions, num_heads=mae.decoder_num_heads).to(device)
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        Defines the forward pass of the SiamMAE.

        Parameters:
        x (torch.Tensor): The input data.

        Returns:
        loss (torch.Tensor): The loss value.
        pred (torch.Tensor): The predicted next frame.
        frame_next_mask (torch.Tensor): The mask applied to the next frame.
        """
        # Split the input into two consecutive frames
        frame_one = x[:, 0, :, :, :]
        frame_next = x[:, 1, :, :, :]

        # Pass the first frame through the MAE's encoder with no masking
        frame_one_x, _, _ = self.mae.forward_encoder(frame_one, mask_ratio = 0)
        frame_one_x = frame_one_x[:, 1:, :]

        # Pass the next frame through the MAE's encoder with the specified mask ratio
        frame_next_x, frame_next_mask, frame_next_ids = self.mae.forward_encoder(frame_next, mask_ratio =self.mask_ratio)

        # Use the CrossDecoder to attend the tokens from the first frame to the tokens from the next frame
        cross_decoded = self.cross_decoder(frame_one_x, frame_next_x)

        # Pass the cross decoded tokens and the masked positions through the MAE's decoder to generate the predicted next frame
        pred = self.mae.forward_decoder(cross_decoded, frame_next_ids)

        # Calculate the loss using the MAE's loss function
        loss = self.mae.forward_loss(frame_next, pred, frame_next_mask)

        return loss, pred, frame_next_mask
    
def prepare_model(chkpt_dir = 'checkpoints/mae_pretrained.pth', arch='mae_vit_base_patch16'):
    """
    Prepares the model for use by building it and loading the weights from a checkpoint.

    Parameters:
    chkpt_dir (str): The directory where the checkpoint file is located.
    arch (str): The architecture of the model to be built.

    Returns:
    model (torch.nn.Module): The prepared model.
    """
    # Build the model using the specified architecture
    model = getattr(models_mae, arch)()

    # Load the weights from the checkpoint file
    checkpoint = torch.load(chkpt_dir, map_location='cpu')

    # Load the state dict from the checkpoint into the model
    # The strict=False argument allows for a partial match between the model and checkpoint state dict
    msg = model.load_state_dict(checkpoint['model'], strict=False)

    # Print any mismatched keys
    print(msg)

    return model

def download_pretrained_model(model_url="https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth", chkpt_dir = 'checkpoints/mae_pretrained.pth'):
    """
    Downloads a pretrained model from a specified URL and saves it to a specified directory.

    Parameters:
    model_url (str): The URL where the pretrained model can be downloaded.
    chkpt_dir (str): The directory where the downloaded model should be saved.

    Returns:
    None
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(chkpt_dir), exist_ok=True)

    # Only download the model if it doesn't already exist in the directory
    if not os.path.exists(chkpt_dir): 
        # Send a GET request to the model URL
        r = requests.get(model_url)
        # Write the content of the response to a file in the specified directory
        with open(chkpt_dir, 'wb') as f:
            f.write(r.content)