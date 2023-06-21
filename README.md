# Upgradable Multimodal Intelligence
Code for CVPR 23 [Towards Fast Adaptation of Pretrained Contrastive Models for Multi-channel Video-Language Retrieval](https://arxiv.org/abs/2206.02082)

This sample implmentation is largely based on the codebase of [Just Ask](https://antoyang.github.io/just-ask.html). So please refer to it for setting up the environment. We will release the full codebase soon after CVPR.

## After setting up, using the files under How2QA can obtain results in Table 1 for the Text+Text variant using the following command

python main_videoqa.py --checkpoint_dir=LOCATION_OF_EXPERIMENT --dataset=how2qa --lr=0.00005 --mlm_prob 0. --qmax_words 120 --baseline to  --lm all-mpnet-base-v2    --epochs 30 

