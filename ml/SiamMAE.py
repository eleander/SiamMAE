import torch
import os 
import requests
from dataset_loader.dataset import get_kinetics_dataset, unnormalize_for_plot, unnormalize
from mae import models_mae


# A cross decoder consists of decoder blocks with only cross-attention layers, where tokens from f2
# attend to the tokens from f1.
class CrossDecoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, f1, f2):
        # f1: ([BATCH_SIZE, 196, N])
        # f2: ([BATCH_SIZE, 50, N])

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
    def __init__(self, mae, mask_ratio = 0.75, device=torch.device("cuda")):
        super().__init__()
        self.mae = mae 
        self.cross_decoder = CrossDecoder(embed_dim=mae.decoder_embed_dimensions, num_heads=mae.decoder_num_heads).to(device)
        self.mask_ratio = mask_ratio

    def forward(self, x):
        frame_one = x[:, 0, :, :, :]
        frame_next = x[:, 1, :, :, :]

        frame_one_x, _, _ = self.mae.forward_encoder(frame_one, mask_ratio = 0)
        frame_one_x = frame_one_x[:, 1:, :]

        frame_next_x, frame_next_mask, frame_next_ids = self.mae.forward_encoder(frame_next, mask_ratio =self.mask_ratio)

        cross_decoded = self.cross_decoder(frame_one_x, frame_next_x)

        pred = self.mae.forward_decoder(cross_decoded, frame_next_ids)
        loss = self.mae.forward_loss(frame_next, pred, frame_next_mask)
        return loss, pred, frame_next_mask
    
def prepare_model(chkpt_dir = 'checkpoints/mae_pretrained.pth', arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def download_pretrained_model(model_url="https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth", chkpt_dir = 'checkpoints/mae_pretrained.pth'):
    os.makedirs(os.path.dirname(chkpt_dir), exist_ok=True)

    if not os.path.exists(chkpt_dir): 
        r = requests.get(model_url)
        with open(chkpt_dir, 'wb') as f:
            f.write(r.content)