# KGFMCL: Knowledge Graph Fusion View Multi-graph Contrastive Learning for Recommendation

This is our PyTorch implementation for the paper:

> Guangliang He , Zhen Zhang , He Zhenyu , Xie Hao. KGFMCL: Knowledge Graph Fusion View Multi-graph Contrastive Learning for Recommendation. in IEEE Transactions on Knowledge and Data Engineering.

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

- pytorch == 1.5.0
- numpy == 1.15.4
- scipy == 1.1.0
- sklearn == 0.20.0
- torch_scatter == 2.0.5
- networkx == 2.5

## Reproducibility & Example to Run the Codes

To demonstrate the reproducibility of the best performance reported in our paper and faciliate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (might be different for the custormized datasets) in the scripts.

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). 

- Last-fm dataset

```
nohup python main.py --dataset last-fm  --user_neibor_size 64 --item_neibor_size 8  --gamma 1 --alpha 1 --ssl_temp 0.15 --ssl_reg 1e-6 --info_reg 1e-6 --kg_reg 1e-5 --context_hops 3 --gpu_id 0  --batch_size 2048 --l2 1e-05 >last-fmu64i8_g1_a1_t0.15_s_1e-6_i_1e-6_k_1e-5_l2_1e-05_batch_size_2048.txt 2>&1 &
```

- Yelp2018 dataset

```
nohup python main.py --dataset yelp2018 --user_neibor_size 32 --item_neibor_size 64 --gamma 1 --alpha 1 --ssl_temp 0.1 --ssl_reg 1e-6 --info_reg 1e-6 --kg_reg 1e-6 --context_hops 3 --gpu_id 0 --batch_size 2048 --l2 1e-05 >yelp2018u32i64_g1_a1_t0.1_s_1e-6_i_1e-6_k_1e-6_l2_1e-05_batch_size_2048.txt 2>&1 &
```

- Alibaba-iFashion dataset

```
nohup python main.py --dataset alibaba-fashion  --user_neibor_size 64 --item_neibor_size 64  --gamma 1 --alpha 1 --ssl_temp 0.1 --ssl_reg 1e-6 --info_reg 1e-6 --kg_reg 1e-6 --context_hops 3 --gpu_id 0  --batch_size 2048 --l2 1e-04 >alibaba-fashionu64i64_g1_a1_t0.1_s_1e-6_i_1e-6_k_1e-6_l2_1e-04_batch_size_2048.txt 2>&1 &
```

