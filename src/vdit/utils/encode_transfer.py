"""
Encoder output packing/unpacking utilities.
Serializes encode() output to bytes for network transfer or IPC.
"""
import io
import torch


def pack_enc_out(enc_out: dict) -> bytes:
    """
    Pack encode() output dict into bytes for network transfer or IPC.
    Moves all tensors to CPU before saving (device-independent).

    Args:
        enc_out: dict from encode(), contains cond_dit, cond_dec, stats_mean, stats_std, ph, pw, etc.

    Returns:
        bytes: serialized binary data
    """
    # Move all tensors to CPU to avoid GPU binding
    cpu_dict = {}
    for k, v in enc_out.items():
        if torch.is_tensor(v):
            cpu_dict[k] = v.cpu()
        else:
            cpu_dict[k] = v
    
    buffer = io.BytesIO()
    torch.save(cpu_dict, buffer)
    return buffer.getvalue()


def unpack_enc_out(blob: bytes, device: torch.device) -> dict:
    """
    Restore encode() output from bytes and move tensors to the target device.

    Args:
        blob: bytes returned by pack_enc_out()
        device: target device, e.g. torch.device("cuda:1")

    Returns:
        dict: restored enc_out with all tensors on the specified device
    """
    buffer = io.BytesIO(blob)
    enc_out = torch.load(buffer, map_location="cpu")
    
    # Move to target device
    for k, v in enc_out.items():
        if torch.is_tensor(v):
            enc_out[k] = v.to(device)
    
    return enc_out

