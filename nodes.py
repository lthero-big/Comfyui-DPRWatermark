import torch
import os
import sys
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from scipy.stats import norm
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import logging
from datetime import datetime
from scipy.stats import norm, kstest

# 日志工具类
class Loggers:
    """日志管理类，单例模式"""
    _logger = None

    @classmethod
    def get_logger(cls, log_dir: str = './logs') -> 'logging.Logger':
        if cls._logger is None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log')
            cls._logger = logging.getLogger('DPRW_Engine')
            cls._logger.setLevel(logging.INFO)
            cls._logger.handlers.clear()
            formatter = logging.Formatter("%(asctime)s %(levelname)s: [%(name)s] %(message)s")
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            console_handler = logging.StreamHandler()
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)
            cls._logger.addHandler(console_handler)
        return cls._logger

# 设置 ComfyUI 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

MAX_RESOLUTION = 8192

def set_random_seed(seed: int) -> None:
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def choose_watermark_length(total_blocks_needed: int) -> int:
    """根据可用块数选择水印长度"""
    if total_blocks_needed >= 256 * 32:
        return 256
    elif total_blocks_needed >= 128 * 32:
        return 128
    elif total_blocks_needed >= 64 * 32:
        return 64
    return 32

def validate_hex(hex_str: str, expected_length: int, default: bytes) -> bytes:
    """验证并生成十六进制字符串，若无效则返回默认值"""
    if hex_str and len(hex_str) == expected_length and all(c in '0123456789abcdefABCDEF' for c in hex_str):
        return bytes.fromhex(hex_str)
    return default

def common_ksampler(model, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, positive, negative, latent,
                    denoise: float = 1.0, disable_noise: bool = False, start_step = None, last_step= None,
                    force_full_denoise: bool = False, use_dprw: bool = False, watermarked_latent_noise=None):
    """
    通用的 KSampler 函数，处理潜在表示采样并支持 DPRW 水印噪声
    """
    latent_image = latent["samples"]
    # 如果latent_image的shape是(1,4,x,x)，则需要进行fix，它会将(1,4,x,x)的噪声，转换为(1,16,x,x)的噪声（但由于传入进来的是empty_latent,所以直接将维度扩大即可
    # 当如果是(1,16,x,x)），于是就可以不处理了，不过这代码可以保留，它不会影响
    if latent_image.shape[1] == 4:
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if use_dprw and watermarked_latent_noise is not None:
        noise = watermarked_latent_noise["samples"]
    elif disable_noise:
        noise = torch.zeros_like(latent_image, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out,)

class DPRWatermark:
    """DPRW 水印算法的核心实现"""
    def __init__(self, key_hex: str, nonce_hex: str,latent_channels:int=4,  device: str = "cuda",log_dir: str = './logs'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_channels = latent_channels
        self.key = validate_hex(key_hex, 64, os.urandom(32))
        self.nonce = validate_hex(nonce_hex, 32, os.urandom(16))
        self.logger = Loggers.get_logger(log_dir)
        self.logger.info(f"====================DPRW Watermark Begin====================")
        self.logger.info(f"Initialized - Key: {self.key.hex()}")
        self.logger.info(f"Initialized - Nonce: {self.nonce.hex()}")

    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> bytes:
        """生成水印字节"""
        length_bits = message_length if message_length > 0 else choose_watermark_length(total_blocks)
        length_bytes = length_bits // 8
        msg_bytes = message.encode('utf-8')
        padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        repeats = total_blocks // length_bits
        self.logger.info(f"Create watermark - Message: {message}")
        self.logger.info(f"Create watermark - Message Length: {message_length}")
        self.logger.info(f"Create watermark - Watermark repeats: {repeats} times")
        return padded_msg * repeats + b'\x00' * ((total_blocks % length_bits) // 8)

    def _encrypt(self, watermark: bytes) -> str:
        """加密水印并转换为二进制字符串"""
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(watermark) + encryptor.finalize()
        return ''.join(format(byte, '08b') for byte in encrypted)

    def _binarize_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """将噪声二值化"""
        return (torch.sigmoid(noise.to(self.device)) > 0.5).to(torch.uint8).flatten()

    def _embed_bits(self, binary: torch.Tensor, bits: str, window_size: int) -> torch.Tensor:
        """在 GPU 上嵌入水印位"""
        num_windows = len(binary) // window_size
        bits_tensor = torch.tensor([int(b) for b in bits[:num_windows]], dtype=torch.uint8, device=self.device)
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        window_sums = windows.sum(dim=1) % 2
        flip_mask = window_sums != bits_tensor
        mid_idx = window_size // 2
        flip_indices = (torch.arange(num_windows, device=self.device) * window_size + mid_idx)[flip_mask]
        binary[flip_indices] = 1 - binary[flip_indices]
        return binary
    
    def _Gaussian_test(self, noise: torch.Tensor) -> bool:
        if isinstance(noise, torch.Tensor):
            noise = noise.cpu().numpy()
        samples = noise.flatten()
        _, p_value = kstest(samples, 'norm', args=(0, 1))
        if np.isnan(samples).any() or np.isinf(samples).any():
            raise ValueError("Restored noise contains NaN or Inf values")
        if np.var(samples) == 0:
            raise ValueError("Restored noise variance is 0")
        if p_value < 0.05:
            raise ValueError(f"Restored noise failed Gaussian test (p={p_value:.4f})")
        self.logger.info(f"Gaussian test passed: p={p_value:.4f}")
        return True

    def _restore_noise(self, binary: torch.Tensor, shape: tuple, seed: int,original_noise:torch.Tensor) -> torch.Tensor:
        """还原高斯噪声"""
        # set_random_seed(seed)
        # noise = torch.randn(shape, device=self.device)
        binary_reshaped = binary.view(shape[1:])
        original_binary = (torch.sigmoid(original_noise) > 0.5).to(torch.uint8)
        mask = binary_reshaped != original_binary
        u = torch.rand_like(original_noise) * 0.5
        theta = u + binary_reshaped.float() * 0.5
        adjustment = torch.erfinv(2 * theta[mask] - 1) * torch.sqrt(torch.tensor(2.0, device=self.device))
        noise=original_noise.clone()
        noise[mask] = adjustment
        # 高斯检验
        self._Gaussian_test(noise)
        return noise

    # 添加一个对noise进行处理的函数
    def _noise_foo(self,noise: torch.Tensor, )->torch.Tensor:
        if self.latent_channels == 16 and noise.size(1) != 16:
            noise = torch.randn(1, self.latent_channels, noise.size(2), noise.size(3), device=self.device, dtype=noise.dtype)
            self.logger.warning(f"Embed watermark - Using NEW random noise: {noise.shape}")
        elif self.latent_channels == 4:
            # 先判断噪声是否是空值，如果是则重新生成
            samples = noise.cpu().numpy().flatten()
            _, p_value = kstest(samples, 'norm', args=(0, 1))
            if np.var(samples) == 0 or p_value < 0.05: 
                self.logger.warning(f"p_value {p_value}")
                self.logger.warning("Embed watermark - Noise variance is 0 or p-value<0.05 using new random noise")
                noise=torch.randn(1,noise.size(1),noise.size(2),noise.size(3),device=self.device,dtype=noise.dtype)
                self.logger.warning(f"Embed watermark - Using NEW random noise: {noise.shape}")
            else:
                self.logger.info(f"Embed watermark - The noise is not empty and can be used directly")
        return noise

    def embed_watermark(self, noise: torch.Tensor, message: str, message_length: int, window_size: int, seed: int) -> torch.Tensor:
        """嵌入水印到噪声中"""

        self.logger.info(f"====================DPRW Watermark Embedding Begin====================")
        noise = self._noise_foo(noise)
        self.logger.info(f"Embed watermark - Noise shape: {noise.shape}")
        total_blocks = noise.numel() // (noise.shape[0] * window_size)
        self.logger.info(f"Embed watermark - Total blocks: {total_blocks}")
        watermark = self._create_watermark(total_blocks, message, message_length)
        encrypted_bits = self._encrypt(watermark)
        binary = self._binarize_noise(noise)
        binary_embedded = self._embed_bits(binary, encrypted_bits, window_size)
        restore_noise = self._restore_noise(binary_embedded, noise.shape, seed,noise)
        self.logger.info(f"restore_noise.shape {restore_noise.shape}")
        self.logger.info(f"====================DPRW Watermark Embedding End====================")
        return restore_noise

    def extract_watermark(self, noise: torch.Tensor, message_length: int, window_size: int) -> tuple[str, str]:
        """从噪声中提取水印"""
        self.logger.info(f"====================DPRW Watermark Extract Begin====================")
        binary = self._binarize_noise(noise)
        num_windows = len(binary) // window_size
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        bits = windows.sum(dim=1) % 2
        bit_str = ''.join(bits.cpu().numpy().astype(str))
        byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decrypted = cipher.decryptor().update(byte_data) + cipher.decryptor().finalize()
        all_bits = ''.join(format(byte, '08b') for byte in decrypted)
        segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
        msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' for i in range(message_length))
        msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
        self.logger.info(f"====================DPRW Watermark Extract End====================")
        return msg_bin, msg

        # 水印准确性评估
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str, extracted_msg_str:str="") -> float:
        """计算位准确率"""
        self.logger.info(f"====================DPRW Watermark Evaluate Begin====================")
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        min_len = min(len(orig_bin), len(extracted_bin))
        orig_bin, extracted_bin = orig_bin[:min_len], extracted_bin[:min_len]
        accuracy = sum(a == b for a, b in zip(orig_bin, extracted_bin)) / min_len
        self.logger.info(f"Evaluation - Original binary: {orig_bin}")
        self.logger.info(f"Evaluation - Extracted binary: {extracted_bin}")
        self.logger.info(f"Evaluation - Extracted binary length: {len(extracted_bin)}")
        if accuracy > 0.9:
            self.logger.info(f"Evaluation - Extracted message: {extracted_msg_str}")
        self.logger.info(f"Evaluation - Bit accuracy: {accuracy}")
        self.logger.info(f"====================DPRW Watermark Evaluate End====================")
        return accuracy

class DPRLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "init_latent": ("LATENT",),
                "use_seed": ("INT", {"default": 1, "min": 0, "max": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
                "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
                "message": ("STRING", {"default": "lthero"}),
                "latent_channels": ("INT", {"default": 4, "min": 4, "max": 16,"step": 12}),
                "window_size": ("INT", {"default": 1, "min": 1, "max": 5}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "create_watermarked_latents"
    CATEGORY = "DPRW/latent"

    def create_watermarked_latents(self, init_latent, use_seed, seed,  key, nonce, message,latent_channels, window_size):
        """创建带水印的潜在噪声"""
        if not isinstance(init_latent, dict) or "samples" not in init_latent:
            raise ValueError("init_latent must be a dictionary containing 'samples' key")
        # print(init_latent)
        init_noise = init_latent["samples"]
        
        # print(init_noise.shape)
        dprw = DPRWatermark(key, nonce,latent_channels)
        if use_seed:
            set_random_seed(seed)
        message_length = len(message) * 8
        watermarked_noise = dprw.embed_watermark(init_noise, message, message_length, window_size, seed)
        # print(f"watermarked_noise.shape {watermarked_noise.shape}")
        # print(watermarked_noise)
        return ({"samples": watermarked_noise},)

class DPRExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "key": ("STRING", {"default": "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"}),
                "nonce": ("STRING", {"default": "05072fd1c2265f6f2e2a4080a2bfbdd8"}),
                "message": ("STRING", {"default": "lthero"}),
                "window_size": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING","LATENT")  # 返回二进制字符串和解码消息
    FUNCTION = "extract"
    CATEGORY = "DPRW/extractor"

    def extract(self, latents, key, nonce,message, window_size):
        """从潜在表示中提取水印"""
        if not isinstance(latents, dict) or "samples" not in latents:
            raise ValueError("latents must be a dictionary containing 'samples' key")
        
        noise = latents["samples"]
        print(f"noise.shape {noise.shape}")
        dprw = DPRWatermark(key, nonce)
        message_length = len(message) * 8
        extracted_msg_bin, extracted_msg_str = dprw.extract_watermark(noise, message_length, window_size)
        dprw.evaluate_accuracy(message, extracted_msg_bin,extracted_msg_str)
        return (extracted_msg_bin, extracted_msg_str,latents)

class DPRKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "use_dprw_noise": (["enable", "disable"],),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "watermarked_latent_noise": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DPRW/sampling"
    
    def sample(self, model, use_dprw_noise, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, watermarked_latent_noise, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        """高级采样器，支持 DPRW 水印噪声"""
        force_full_denoise = return_with_leftover_noise != "enable"
        use_dprw = use_dprw_noise == "enable"
        disable_noise = add_noise == "disable"
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                               force_full_denoise=force_full_denoise, use_dprw=use_dprw, watermarked_latent_noise=watermarked_latent_noise)

NODE_CLASS_MAPPINGS = {
    "DPR_Latent": DPRLatent,
    "DPR_Extractor": DPRExtractor,
    "DPR_KSamplerAdvanced": DPRKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DPR_Latent": "DPR Latent",
    "DPR_Extractor": "DPR Extractor",
    "DPR_KSamplerAdvanced": "DPR KSampler Advanced",
}
