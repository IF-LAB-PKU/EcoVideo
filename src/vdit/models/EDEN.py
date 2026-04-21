import torch
from torch import nn
from functools import partial
from vdit.modules import VAEBlock, DiTBlock
from vdit.utils import TimestepEmbedder, get_pos_embedding, preprocess_cond
from einops import rearrange


class EDEN(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, patch_size=16, latent_dim=16, hidden_dim=768,
                 num_heads=12, mlp_ratio=4.0, dit_depth=12, decoder_depth=4, qkv_bias=False, attn_drop_rate=0.,
                 proj_drop_rate=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_xformers=True,
                 add_attn_decoder=True, add_attn_type="temporal_attn"):
        super().__init__()
        self.dim = hidden_dim
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        self.patch_cond_dit = nn.Conv2d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_cond_dit = norm_layer(hidden_dim)
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, proj_drop_rate, act_layer, norm_layer,
                     use_xformers) for _ in range(dit_depth)
        ])
        self.norm_out = norm_layer(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, 2 * latent_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        self.difference_embedder = nn.Linear(1, hidden_dim)
        self.denoise_timestep_embedder = TimestepEmbedder(hidden_dim)
        self.patch_cond_dec = nn.Conv2d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_cond_dec = norm_layer(hidden_dim)
        self.proj_token = nn.Linear(latent_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            VAEBlock(hidden_dim, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, proj_drop_rate, act_layer, norm_layer,
                     use_xformers, is_encoder=False, add_attn_decoder=add_attn_decoder, add_attn_type=add_attn_type)
            for _ in range(decoder_depth)
        ])
        final_dim = patch_size * patch_size * out_dim
        self.unpatchify = nn.Sequential(norm_layer(hidden_dim),
                                        nn.Linear(hidden_dim, final_dim),
                                        act_layer(),
                                        nn.Linear(final_dim, final_dim))
        self.stats, self.ph, self.pw = None, None, None
        self.cond_dec = None
        self.pos_embedding = None
        self.query_pos_embedding = None

    def encode(self, cond_frames):
        """
        Encoder方法：将条件帧编码为tokens
        
        Args:
            cond_frames: [2, 3, H', W']，经过padder处理后的两帧拼接结果
        
        Returns:
            dict: 包含编码后的所有信息
                - cond_dit: [2, ph*pw, dim] - 给DiT用的tokens
                - cond_dec: [2, ph*pw, dim] - 给decoder用的tokens
                - stats_mean: 归一化的均值
                - stats_std: 归一化的标准差
                - ph: patch高度数量
                - pw: patch宽度数量
        """
        # 1. 归一化并记录统计信息
        x, stats = preprocess_cond(cond_frames)
        self.stats = stats  # 为decode()做准备
        
        # 2. 计算patch网格尺寸
        self.ph = x.shape[-2] // self.patch_size
        self.pw = x.shape[-1] // self.patch_size
        
        # 3. 位置编码
        self.pos_embedding = get_pos_embedding(self.ph, self.pw, 1, self.dim).to(x.device)
        
        # 4. DiT分支编码
        cond_dit = self.patch_cond_dit(x).flatten(2).transpose(1, 2)  # [2, ph*pw, dim]
        cond_dit = self.norm_cond_dit(cond_dit + self.pos_embedding)
        
        # 5. Decoder分支编码
        cond_dec = self.patch_cond_dec(x).flatten(2).transpose(1, 2)  # [2, ph*pw, dim]
        cond_dec = self.norm_cond_dec(cond_dec + self.pos_embedding)
        
        # 6. 保存给decode()用
        self.cond_dec = cond_dec
        
        # 7. 返回编码结果
        return {
            "cond_dit": cond_dit,      # [2, ph*pw, dim]
            "cond_dec": cond_dec,      # [2, ph*pw, dim]
            "stats_mean": stats[0],    # 归一化均值
            "stats_std": stats[1],     # 归一化标准差
            "ph": self.ph,             # patch高度数量
            "pw": self.pw,             # patch宽度数量
        }

    def patch_cond(self, x):
        x, self.stats = preprocess_cond(x)
        self.pos_embedding = get_pos_embedding(self.ph, self.pw, 1, self.dim).to(x.device)
        x_dit = self.patch_cond_dit(x).flatten(2).transpose(1, 2)
        x_dit = self.norm_cond_dit(x_dit + self.pos_embedding)
        x_dec = self.patch_cond_dec(x).flatten(2).transpose(1, 2)
        x_dec = self.norm_cond_dec(x_dec + self.pos_embedding)
        return x_dit, x_dec

    def postprocess(self, x):
        return x * self.stats[1] + self.stats[0]

    def pixel_shuffle(self, x):
        x = self.unpatchify(x)
        x = rearrange(x, "b (ph pw) (p1 p2 c) -> b c (ph p1) (pw p2)",
                      ph=self.ph, pw=self.pw, p1=self.patch_size, p2=self.patch_size, c=self.out_dim)
        return x

    def decode(self, query_latents):
        tokens_0, tokens_1 = self.cond_dec.chunk(2, dim=0)
        cond_tokens = torch.cat((tokens_0, tokens_1), dim=1)
        query_tokens = self.proj_token(query_latents)
        query_frames = rearrange(query_tokens, "b (ph pw) d -> b d ph pw", ph=self.ph // 2, pw=self.pw // 2)
        recon_frames = nn.functional.interpolate(query_frames, scale_factor=2., mode="bicubic", align_corners=False)
        recon_tokens = recon_frames.flatten(2).transpose(1, 2) + self.pos_embedding
        query_tokens = query_tokens + self.query_pos_embedding
        for blk in self.decoder_blocks:
            recon_tokens = blk(query_tokens, recon_tokens, cond_tokens, self.ph, self.pw)
        recon_frames = self.postprocess(self.pixel_shuffle(recon_tokens))
        return recon_frames

    def denoise(self, query_latents, denoise_timestep, cond_frames, difference):
        self.ph, self.pw = cond_frames.shape[-2] // 16, cond_frames.shape[-1] // 16
        cond_dit, self.cond_dec = self.patch_cond(cond_frames)
        tokens_0, tokens_1 = cond_dit.chunk(2, dim=0)
        denoise_timestep_embedding = self.denoise_timestep_embedder(denoise_timestep)
        difference_embedding = self.difference_embedder(difference)
        condition_embedding = denoise_timestep_embedding + difference_embedding
        modulations = self.adaLN_modulation(condition_embedding)
        self.query_pos_embedding = get_pos_embedding(self.ph, self.pw, 2, self.dim).to(query_latents.device)
        query_embedding = self.proj_in(query_latents) + self.query_pos_embedding
        for blk in self.dit_blocks:
            query_embedding = blk(query_embedding, tokens_0, tokens_1, self.ph, self.pw, modulations)
        query_latents = self.proj_out(self.norm_out(query_embedding))
        query_latents, _ = query_latents.chunk(2, dim=-1)
        return query_latents

    def denoise_from_tokens(self, query_latents, denoise_timestep, enc_out, difference):
        """
        使用encoder输出的tokens进行去噪（不重新调用patch_cond）
        
        Args:
            query_latents: [1, num_patches, latent_dim] - 当前扩散状态z_t
            denoise_timestep: [1] - 当前时间步t
            enc_out: dict - encode()的输出，包含cond_dit, cond_dec, stats等
            difference: [1, 1] - 余弦相似度embedding的输入
        
        Returns:
            query_latents: [1, num_patches, latent_dim] - 去噪后的latent
        """
        # 1. 从enc_out中取出encoder输出
        cond_dit = enc_out["cond_dit"]      # [2, ph*pw, dim]
        cond_dec = enc_out["cond_dec"]      # [2, ph*pw, dim]
        stats_mean = enc_out["stats_mean"]
        stats_std = enc_out["stats_std"]
        ph = enc_out["ph"]
        pw = enc_out["pw"]
        
        # 2. 更新self的状态（让decode能用到）
        self.ph, self.pw = ph, pw
        self.stats = (stats_mean, stats_std)
        self.cond_dec = cond_dec
        # 设置pos_embedding（decode()需要用到）
        self.pos_embedding = get_pos_embedding(ph, pw, 1, self.dim).to(query_latents.device)
        
        # 3. 生成时间步和差异的embedding
        denoise_timestep_embedding = self.denoise_timestep_embedder(denoise_timestep)
        difference_embedding = self.difference_embedder(difference)
        condition_embedding = denoise_timestep_embedding + difference_embedding
        modulations = self.adaLN_modulation(condition_embedding)
        
        # 4. Query位置编码
        self.query_pos_embedding = get_pos_embedding(ph, pw, 2, self.dim).to(query_latents.device)
        query_embedding = self.proj_in(query_latents) + self.query_pos_embedding
        
        # 5. DiT主干处理
        tokens_0, tokens_1 = cond_dit.chunk(2, dim=0)
        for blk in self.dit_blocks:
            query_embedding = blk(query_embedding, tokens_0, tokens_1, ph, pw, modulations)
        
        # 6. 输出投影
        query_latents = self.proj_out(self.norm_out(query_embedding))
        query_latents, _ = query_latents.chunk(2, dim=-1)
        return query_latents

