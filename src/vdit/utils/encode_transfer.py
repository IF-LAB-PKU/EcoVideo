"""
Encoder输出打包/解包工具函数
用于将encode()的输出打包成bytes，方便网络传输或进程间传递
"""
import io
import torch


def pack_enc_out(enc_out: dict) -> bytes:
    """
    将 encode() 输出的 dict 打包成 bytes，方便网络传输或进程间传递。
    会把所有 tensor 移到 CPU 再保存（这样不依赖某个 GPU）。
    
    Args:
        enc_out: encode()返回的dict，包含cond_dit, cond_dec, stats_mean, stats_std, ph, pw等
    
    Returns:
        bytes: 打包后的二进制数据
    """
    # 先把所有 tensor 移到 CPU，避免绑定到某个 GPU
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
    从 bytes 还原 encode() 输出，并把里面的 tensor 移到指定 device（比如 cuda:1）。
    
    Args:
        blob: pack_enc_out()返回的bytes数据
        device: 目标设备，如torch.device("cuda:1")
    
    Returns:
        dict: 还原后的enc_out，所有tensor都在指定device上
    """
    buffer = io.BytesIO(blob)
    enc_out = torch.load(buffer, map_location="cpu")
    
    # 移到目标 device
    for k, v in enc_out.items():
        if torch.is_tensor(v):
            enc_out[k] = v.to(device)
    
    return enc_out

