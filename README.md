# Multi-granularity Context Perception Network for Open Set Recognition of Camouflaged Objects
> **Authors:** 
> [*Ze Song*](https://scholar.google.com/citations?user=uatSii8AAAAJ&hl=zh-CN&oi=sra),
> [*Xudong Kang*](https://scholar.google.com/citations?user=5XOeLZYAAAAJ&hl=en),
> [*Xiaohui Wei*](https://scholar.google.co.il/citations?user=Uq50h3gAAAAJ&hl=zh-CN),
> [*Renwei Dian*](https://scholar.google.com/citations?user=EoTrH5UAAAAJ&hl=en),
> [*Jinyang Liu*](https://scholar.google.com/citations?user=PxUXOdsAAAAJ&hl=en),
> and [*Shutao Li*](https://scholar.google.com/citations?user=PlBq8n8AAAAJ&hl=en).


Code implementation of "_**Multi-granularity Context Perception Network for Open Set Recognition of Camouflaged Objects**_". 

ACOC download link [Google Drive](https://drive.google.com/file/d/14dwo37hSMz-gjRPpnLLVo2LVl4_bl95j/view?usp=drive_link). 
NCOC download link [Google Drive](https://drive.google.com/file/d/1LgToD8QQRJ6AelA2VC0dqkIwOlQcXxBw/view?usp=drive_link).
## Usage
### 1. Download pre-trained ViT model
Please download model from the official websites: 
* [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) or [baidu](https://pan.baidu.com/share/init?surl=vwJxnJcVqcLZAw9HaqiR6g) with the fetch code: swin.
* move it into ``` ./pretrained_ckpt/ ```

### 2. Train

To train FSNet with costumed path:

```bash
python MyTrain_Val.py --save_path './snapshot/FSNet/'
```
### 3. Test

To test with trained model:

```bash
python MyTesting.py --pth_path './snapshot/FSNet/Net_epoch_best.pth'
```

downloading our weights and move it into `./snapshot/FSNet/`, 
    which can be found from [Google Drive](https://drive.google.com/file/d/1Bgi8MThe1eEE9fYyaHuLHacO1Cs_e9Ta/view?usp=share_link).
    
 You can also download prediction maps from [Google Drive](https://drive.google.com/file/d/1kT9l9NrwWCffP4EQ7ITBd834389xY8iV/view?usp=share_link).





