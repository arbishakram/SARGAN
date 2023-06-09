# SARGAN: Spatial Attention-based Residuals for Facial Expression Manipulation, (TCSVT, 2023)

<p align="center"><img width="100%" src="imgs/sargan_teaser.png" /></p>

This repository provides the official implementation of the following paper:
>[SARGAN: Spatial Attention-based Residuals for Facial Expression Manipulation](https://arxiv.org/abs/2303.17212)<br>
> [Arbish Akram](https://arbishakram.github.io/) and [Nazar Khan](http://faculty.pucit.edu.pk/nazarkhan/) <br>
> Department of Computer Science, University of the Punjab, Lahore, Pakistan.<br>
> In IEEE Transactions on Circuits and Systems for Video Technology (TCSV), 2023


> **Abstract:** *Encoder-decoder based architecture has been widely used in the generator of generative adversarial networks for facial manipulation. However, we observe that the current architecture fails to recover the input image color, rich facial details such as skin color or texture and introduces artifacts as well. In this paper, we present a novel method named SARGAN that addresses the above-mentioned limitations from three perspectives. First, we employed spatial attention-based residual block instead of vanilla residual blocks to properly capture the expression-related features to be changed while keeping the other features unchanged. Second, we exploited a symmetric encoder-decoder network to attend facial features at multiple scales. Third, we proposed to train the complete network with a residual connection which relieves the generator of pressure to generate the input face image thereby producing the desired expression by directly feeding the input image towards the end of the generator. Both qualitative and quantitative experimental results show that our proposed model performs significantly better than state-of-the-art methods. In addition, existing models require much larger datasets for training but their performance degrades on out-of-distribution images. In contrast, SARGAN can be trained on smaller facial expressions datasets, which generalizes well on out-of-distribution images including human photographs, portraits, avatars and statues.*

## Test with Pretrained Model
```
python driver.py --mode test --image_size 128 --c_dim 7 --image_dir ./testing_imgs/ --model_save_dir ./pre-trained_model/ \
                 --result_dir ./sargan/results                               
```

## Train the Model
```
python driver.py --mode train --image_size 128 --c_dim 7 --batch_size 8 --image_dir ./dataset/  --model_save_dir ./sargan/models/ \
                 --log_dir ./sargan/logs --sample_dir ./sargan/samples --result_dir ./sargan/results                            
```

## Citation
If you find this work useful for your research, please cite our paper:
```
@article{akram23sargan,
  author={Akram, Arbish and Khan, Nazar},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={SARGAN: Spatial Attention-based Residuals for Facial Expression Manipulation}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3255243}}
```

## Acknowledgement
This code is based on Yunjey's [StarGAN](https://github.com/yunjey/stargan) with modifications.
