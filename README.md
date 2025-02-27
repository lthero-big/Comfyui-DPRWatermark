## 概述

论文："Diversity-Preserving Robust Watermarking for Diffusion Model Generated Images".

论文项目：[https://github.com/lthero-big/DPRW/](https://github.com/lthero-big/DPRW/)

DPRW（Diffusion-based Perceptual Robust Watermarking）节点是为 ComfyUI 设计的自定义节点，旨在通过将水印嵌入到潜在噪声中来保护生成的图像。它利用扩散模型的特性，在不显著影响图像质量的情况下嵌入和提取水印信息。DPRW 节点提供了三个主要功能类，帮助用户在图像生成过程中添加水印、提取水印并进行高级采样。

### 主要功能
- **`DPRLatent`**: 用于创建带水印的潜在噪声。
- **`DPRExtractor`**: 从潜在表示中提取水印。
- **`DPRKSamplerAdvanced`**: 高级采样器，支持使用带水印的潜在噪声生成图像。

---

## 安装

在使用 DPRW 节点之前，您需要安装 ComfyUI 并确保其版本兼容。以下是安装步骤：

### 步骤
1. **进入 ComfyUI 目录**
   在终端或命令行中运行以下命令：
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. **添加 DPRW 节点**
   将包含 DPRW 节点的 Python 文件复制到 ComfyUI 的 `custom_nodes` 目录
   ```
   git clone https://github.com/lthero-big/Comfyui-DPRWatermark.git
   ```

3. **启动 ComfyUI**
   在 ComfyUI 目录下运行：
   ```bash
   python main.py
   ```

---

## 使用案例

`example_workflows` 中包含一个使用FLUX生成DPRW水印图片并进行水印提取的工作流

### 示例工作流

以下是一个完整的工作流示例：

1. **创建带水印噪声**
   - 添加 `EmptyLatentImage` 节点生成初始潜在表示。
   - 添加 `DPR_Latent` 节点，连接初始潜在表示，设置 `message` 为 "MyWatermark"。
   - 输出：带水印的潜在噪声。

2. **生成图像**
   - 添加 `DPR_KSamplerAdvanced` 节点。
   - 连接模型、条件和带水印噪声，设置 `add_dprw_noise` 为 "enable"。
   - 输出：带水印的潜在表示。
   - 使用 `VAEDecode` 节点解码为图像。

3. **提取水印**
   - 添加 `DPRExtractor` 节点，连接采样后的潜在表示。
   - 输入相同的 `key`、`nonce` 和 `window_size`，运行以提取水印。

---

### 参数说明

- **`key`**：256 位密钥（64 个十六进制字符），用于加密水印。
- **`nonce`**：128 位 nonce（32 个十六进制字符），增强加密安全性。
- **`message`**：水印内容，建议简短以确保嵌入效率。
- **`window_size`**：窗口大小影响水印的鲁棒性和容量，建议从小值开始测试。

---

## 设置和使用

DPRW 节点的使用涉及创建带水印的潜在噪声、提取水印以及使用高级采样器生成图像。以下是每个类的详细说明和使用方法。

### 1. DPRLatent - 创建带水印的潜在噪声

**功能**：将水印嵌入到潜在噪声中。

#### 输入参数
| 参数             | 类型    | 默认值                                      | 描述                                   |
|------------------|---------|--------------------------------------------|----------------------------------------|
| `init_latent`    | LATENT  | -                                          | 初始潜在表示（必须是包含样本的字典）   |
| `use_seed`       | INT     | 1 (0 或 1)                                 | 是否使用固定种子 (1 表示是，0 表示否)  |
| `seed`           | INT     | 42                                         | 随机种子，范围 [0, 0xffffffff]         |
| `key`            | STRING  | "5822ff9cce6772f714192f..."               | 256 位加密密钥（64 个十六进制字符）    |
| `nonce`          | STRING  | "05072fd1c2265f6f2e2a40..."               | 128 位加密 nonce（32 个十六进制字符）  |
| `message`        | STRING  | "lthero"                                   | 要嵌入的水印消息                       |
| `latent_channels`| INT     | 4 (4 或 16)                                | 潜在通道数，若非FLUX等模型，只能使用4；FLUX模型必须使用16                             |
| `window_size`    | INT     | 1                             | 水印嵌入窗口大小，影响鲁棒性和容量     |

#### 使用步骤
1. 在 ComfyUI 工作流中添加 `DPR_Latent` 节点。
2. 连接一个初始潜在表示（`LATENT` 类型，例如从 `EmptyLatentImage` 节点生成）。
3. 设置参数：
   - `key` 和 `nonce` 用于加密水印，确保安全性。
   - `message` 是您想嵌入的信息（例如版权声明）。
   - `window_size` 建议从 1 开始，根据需求调整（较大值增强鲁棒性）。
4. 运行工作流，输出为带水印的潜在噪声（`LATENT` 类型）。

#### 输出
- `LATENT`：包含带水印噪声的潜在表示。

---

### 2. DPRExtractor - 提取水印

**功能**：从潜在表示中提取嵌入的水印。

#### 输入参数
| 参数          | 类型    | 默认值                                      | 描述                                   |
|---------------|---------|--------------------------------------------|----------------------------------------|
| `latents`     | LATENT  | -                                          | 包含水印的潜在表示                     |
| `key`         | STRING  | "5822ff9cce6772f714192f..."               | 用于解密的密钥（必须与嵌入时一致）     |
| `nonce`       | STRING  | "05072fd1c2265f6f2e2a40..."               | 用于解密的 nonce（必须与嵌入时一致）   |
| `message`     | STRING  | "lthero"                                   | 原始消息，用于准确性评估               |
| `window_size` | INT     | 1 (1 至 100)                               | 提取时的窗口大小（必须与嵌入时一致）   |

#### 使用步骤
1. 在工作流中添加 `DPR_Extractor` 节点。
2. 连接包含水印的潜在表示（通常来自 `DPRLatent` 或 `DPRKSamplerAdvanced`）。
3. 输入与嵌入时相同的 `key`、`nonce` 和 `window_size`。
4. 运行工作流，提取水印。

#### 输出
- `STRING`：提取的二进制水印（位字符串）。
- `STRING`：解码后的消息（文本字符串）。
- `LATENT`：原始潜在表示（用于进一步处理）。

#### 注意
- 如果 `key` 或 `nonce` 不匹配，将无法正确提取水印。
- 日志会记录提取的准确性（位准确率），通常应大于 90%。

---

### 3. DPRKSamplerAdvanced - 高级采样器

**功能**：使用带水印的潜在噪声生成图像。

#### 输入参数
| 参数                     | 类型         | 默认值                    | 描述                                   |
|--------------------------|--------------|---------------------------|----------------------------------------|
| `model`                  | MODEL        | -                         | 扩散模型                               |
| `add_dprw_noise`         | STRING       | "enable" / "disable"      | 是否使用 DPRW 水印噪声                 |
| `add_noise`              | STRING       | "enable" / "disable"      | 是否添加常规噪声                       |
| `noise_seed`             | INT          | 42                        | 噪声种子，范围 [0, 0xffffffffffffffff] |
| `steps`                  | INT          | 20 (1 至 10000)           | 采样步数                               |
| `cfg`                    | FLOAT        | 8.0 (0.0 至 100.0)        | CFG 值（引导强度）                     |
| `sampler_name`           | STRING       | KSampler.SAMPLERS         | 采样器类型（如 DDIM、Euler 等）        |
| `scheduler`              | STRING       | KSampler.SCHEDULERS       | 调度器类型                             |
| `positive`               | CONDITIONING | -                         | 正面条件                               |
| `negative`               | CONDITIONING | -                         | 负面条件                               |
| `latent_image`           | LATENT       | -                         | 输入潜在图像                           |
| `dprw_latent_noise`      | LATENT       | -                         | DPRW 带水印噪声（从 `DPRLatent` 输出） |
| `start_at_step`          | INT          | 0 (0 至 10000)            | 开始步数                               |
| `end_at_step`            | INT          | 10000 (0 至 10000)        | 结束步数                               |
| `return_with_leftover_noise` | STRING   | "disable" / "enable"      | 是否返回带剩余噪声的图像               |

#### 使用步骤
1. 在工作流中添加 `DPR_KSamplerAdvanced` 节点。
2. 连接必要的输入：
   - `model`：扩散模型（例如 Stable Diffusion 模型）。
   - `latent_image`：初始潜在图像。
   - `dprw_latent_noise`：从 `DPRLatent` 输出的带水印噪声。
   - `positive` 和 `negative`：条件（通常来自文本提示）。
3. 设置参数：
   - 将 `add_dprw_noise` 设为 "enable" 以使用水印噪声。
   - 选择 `sampler_name`（如 "ddim"）和 `scheduler`。
4. 运行工作流，生成带水印的图像。

#### 输出
- `LATENT`：生成的潜在表示，可进一步解码为图像。

---


## 引用

- **[ComfyUI GitHub 页面](https://github.com/comfyanonymous/ComfyUI)** - ComfyUI 官方仓库。
- **[ComfyUI 社区手册](https://blenderneko.github.io/ComfyUI-docs/)** - 社区文档和教程。

通过以上步骤，您可以快速上手 DPRW 节点，为您的图像生成添加水印保护！如果有任何问题，请查阅日志或咨询 ComfyUI 社区。
