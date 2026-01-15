from typing import List, Tuple, Dict
import torch as th
import torch.nn.functional as F
from omegaconf import OmegaConf
from mineclip import MineCLIP
from abc import ABC, abstractstaticmethod
import json
import hashlib


class ClipReward(ABC):
    def __init__(self, ckpt="weights/mineclip_attn.pth", **kwargs) -> None:
        kwargs["arch"] = kwargs.pop("arch", "vit_base_p16_fz.v2.t2")
        kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 512)
        kwargs["image_feature_dim"] = kwargs.pop("image_feature_dim", 512)
        kwargs["mlp_adapter_spec"] = kwargs.pop("mlp_adapter_spec", "v0-2.t0")
        kwargs["pool_type"] = kwargs.pop("pool_type", "attn.d2.nh8.glusw")
        kwargs["resolution"] = [160, 256]

        self.resolution = self.get_resolution()
        self.device = kwargs.pop("device", "cuda")
        self.model = None
        
        self._load_mineclip(ckpt, kwargs)

    @abstractstaticmethod
    def get_resolution():
        raise NotImplementedError()

    @abstractstaticmethod
    def _get_curr_frame(obs):
        raise NotImplementedError()

    def _load_mineclip(self, ckpt, config):
        import os
        global _MINECLIP_SHARED_CACHE
        if "_MINECLIP_SHARED_CACHE" not in globals():
            _MINECLIP_SHARED_CACHE = {}
        
        # Auto-download checkpoint if not found
        if not os.path.exists(ckpt):
            print(f"⚠ MineCLIP checkpoint not found: {ckpt}")
            print(f"  Downloading from Google Drive...")
            self._download_checkpoint(ckpt)
        
        config = OmegaConf.create(config)
        # Share MineCLIP weights across env wrappers / training to avoid duplicate VRAM use.
        # Key is based on ckpt path, device, and MineCLIP constructor config.
        try:
            cfg_dict = OmegaConf.to_container(config, resolve=True)
        except Exception:
            cfg_dict = dict(config)
        key_str = json.dumps({"ckpt": ckpt, "device": str(self.device), "cfg": cfg_dict}, sort_keys=True, default=str)
        cache_key = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
        if cache_key in _MINECLIP_SHARED_CACHE:
            self.model, loaded_ckpt = _MINECLIP_SHARED_CACHE[cache_key]
        else:
            self.model = MineCLIP(**config).to(self.device)
            loaded_ckpt = None
            _MINECLIP_SHARED_CACHE[cache_key] = (self.model, loaded_ckpt)
        
        if os.path.exists(ckpt) and loaded_ckpt != ckpt:
            self.model.load_ckpt(ckpt, strict=True)
            _MINECLIP_SHARED_CACHE[cache_key] = (self.model, ckpt)
            print(f"✓ MineCLIP weights loaded from {ckpt}")
        else:
            if not os.path.exists(ckpt):
                print(f"✗ Download failed! Model will use random weights.")
            
        if self.resolution != (160, 256):  # Not ideal, but we need to resize the relative position embedding
            self.model.clip_model.vision_model._resolution = th.tensor([160, 256])  # This isn't updated from when mineclip resized it
            self.model.clip_model.vision_model.resize_pos_embed(self.resolution)
        self.model.eval()
    
    def _download_checkpoint(self, checkpoint_path):
        """Automatically download MineCLIP checkpoint from Google Drive using gdown."""
        try:
            import subprocess
            import sys
            import os
            
            # Ensure gdown is installed
            try:
                import gdown
            except ImportError:
                print("  Installing gdown...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
                import gdown
            
            # Create weights directory if it doesn't exist
            weights_dir = os.path.dirname(checkpoint_path)
            if weights_dir:
                os.makedirs(weights_dir, exist_ok=True)
                print(f"  Created directory: {weights_dir}")
            
            # Download from Google Drive
            file_id = "1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            print(f"  Downloading MineCLIP checkpoint (~92MB)...")
            gdown.download(url, checkpoint_path, quiet=False)
            
            if os.path.exists(checkpoint_path):
                file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"✓ Download complete! Saved to {checkpoint_path} ({file_size_mb:.1f} MB)")
            else:
                print(f"✗ Download failed - file not found at {checkpoint_path}")
                
        except Exception as e:
            print(f"✗ Failed to download checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def _get_reward_from_logits(
        self,
        logits: th.Tensor  # P
    ) -> float:
        probs = th.softmax(logits, 0)
        return max(probs[0].item() - 1 / logits.shape[0], 0)

    def _get_image_feats(
        self,
        curr_frame: th.Tensor,
        past_frames: th.Tensor = None
    ) -> th.Tensor:
        while len(curr_frame.shape) < 5:
            curr_frame = curr_frame.unsqueeze(0)
        assert curr_frame.shape == (1, 1, 3) + self.resolution, "Found shape {}".format(curr_frame.shape)
        curr_frame_feats = self.model.forward_image_features(curr_frame.to(self.device))  # 1 x 1 x 512

        if past_frames is None:
            # Spec-matching padding: if we don't yet have 15 past frames, left-pad by
            # repeating the earliest available frame (at reset this is curr_frame).
            past_frames = curr_frame_feats.detach().cpu().repeat(1, 15, 1)[0]  # 15 x 512
        past_frames = past_frames.to(self.device)

        while len(past_frames.shape) < 3:
            past_frames = past_frames.unsqueeze(0)
        assert past_frames.shape == (1, 15, curr_frame_feats.shape[-1]), "Found shape {}".format(past_frames.shape)

        return th.cat((past_frames, curr_frame_feats), dim=1)
        
    def _get_video_feats(
        self,
        image_feats: th.Tensor
    ) -> th.Tensor:
        return self.model.forward_video_features(image_feats.to(self.device))  # 1 x 512

    def _get_text_feats(
        self,
        prompts: str
    ) -> th.Tensor:
        text_feats = self.model.encode_text(prompts)  # P x 512
        assert len(text_feats.shape) == 2 and text_feats.shape[0] == len(prompts), "Found shape {}".format(text_feats.shape)
        return text_feats

    def get_logits(
        self,
        obs: Dict,  # 3 x 160 x 256
        prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None,  # history x 512
        return_embeds: bool = False,
        text_select: str = "argmax",  # {"argmax","first"}
    ):
        curr_frame = self._get_curr_frame(obs)
        past_frames, text_feats = state

        with th.no_grad():
            if text_feats is None:
                text_feats = self._get_text_feats(prompts)

            image_feats = self._get_image_feats(curr_frame, past_frames)
            video_feats = self._get_video_feats(image_feats)
            video_feats_d = video_feats.to(self.device)
            text_feats_d = text_feats.to(self.device)
            logits = self.model.forward_reward_head(video_feats_d, text_tokens=text_feats_d)[0][0]  # P

            extra = None
            if return_embeds:
                # Text-conditioned clip embedding for MPF-LSD:
                # mc_e = normalize( normalize(v) ⊙ normalize(u) )
                v = video_feats_d[0]  # 512
                if text_select == "first":
                    idx = 0
                elif text_select == "argmax":
                    idx = int(th.argmax(logits).item()) if logits.numel() else 0
                else:
                    raise ValueError(f"Unknown text_select={text_select}")
                u = text_feats_d[idx]  # 512
                v_n = F.normalize(v.float(), dim=-1)
                u_n = F.normalize(u.float(), dim=-1)
                mc_e = F.normalize(v_n * u_n, dim=-1)
                extra = {
                    "video_feats": video_feats_d.detach().cpu(),   # 1 x 512
                    "text_feats": text_feats_d.detach().cpu(),     # P x 512
                    "text_idx": idx,
                    "mc_e": mc_e.detach().cpu(),                 # 512
                }

        if return_embeds:
            return logits, (image_feats[0, 1:].cpu(), text_feats.cpu()), extra
        else:
            return logits, (image_feats[0, 1:].cpu(), text_feats.cpu())

    def get_reward(
        self, 
        obs: Dict,
        prompt: str,
        neg_prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[float, Tuple[th.Tensor, th.Tensor]]:
        logits, state = self.get_logits(
            obs,
            [prompt] + neg_prompts,
            state
        )
        reward = self._get_reward_from_logits(logits)

        return reward, state

    def get_rewards(
        self, 
        obs: Dict,
        prompts: List[str],
        neg_prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[List[float], Tuple[th.Tensor, th.Tensor]]:
        logits, state = self.get_logits(
            obs,
            prompts + neg_prompts,
            state
        )
        rewards = []
        for i in range(len(prompts)):
            rewards.append(self._get_reward_from_logits(th.cat((
                logits[i:i+1],
                logits[len(prompts):]
            ))))
        return rewards, state
