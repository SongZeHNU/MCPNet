# Multi-granularity Context Perception Network for Open Set Recognition of Camouflaged Objects
> **Authors:** 
> [*Ze Song*](https://scholar.google.com/citations?user=uatSii8AAAAJ&hl=zh-CN&oi=sra),
> [*Xudong Kang*](https://scholar.google.com/citations?user=5XOeLZYAAAAJ&hl=en),
> [*Xiaohui Wei*](https://scholar.google.co.il/citations?user=Uq50h3gAAAAJ&hl=zh-CN),
> [*Renwei Dian*](https://scholar.google.com/citations?user=EoTrH5UAAAAJ&hl=en),
> [*Jinyang Liu*](https://scholar.google.com/citations?user=PxUXOdsAAAAJ&hl=en),
> and [*Shutao Li*](https://scholar.google.com/citations?user=PlBq8n8AAAAJ&hl=en).


Code implementation of "_**Multi-granularity Context Perception Network for Open Set Recognition of Camouflaged Objects**_". 


## Usage

### 1. Environments
Our code is configured by Pytorch 1.9.0, torchvision 0.10.0, Python 3.8.16.

### 2. Datasets
ACOC download link [Google Drive](https://drive.google.com/file/d/14dwo37hSMz-gjRPpnLLVo2LVl4_bl95j/view?usp=drive_link). 

NCOC download link [Google Drive](https://drive.google.com/file/d/1LgToD8QQRJ6AelA2VC0dqkIwOlQcXxBw/view?usp=drive_link).

### 3. Download pre-trained ViT model
Please download model from the official websites: 
* ViT-B16 [Google Drive](https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz).
* move it into ``` ./ ```

### 4. Train

To train MCPNet with costumed path:

```bash
python train.py --dataset 'acoc' --name mcpnet --gpu 0
```
### 5. Test

To test with trained model:

```bash
python test.py --dataset 'acoc' --gpu 0
```






