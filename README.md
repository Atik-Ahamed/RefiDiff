
Welcome to our codebase for **RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation**.

## RefiDiff is accepted to [**AAAI, 2026**](https://aaai.org/conference/aaai/aaai-26/) 

# Environment:
We recommend creating a dedicated Conda environment to ensure compatibility. Please follow the commands below:


```
conda create -n refidiff python=3.12    

conda activate refidiff

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

conda install nvidia/label/cuda-12.4.0::cuda-toolkit

pip install -r requirements/refidiff.txt
```
> Please consider manual installation if any issues arise.

# Preparing Datasets

```
bash scripts/process_data.sh
```

# Running on a dataset


[NAME_OF_DATASET]: example dataset name (e.g., california)

[MASK_IDX]: example mask id (e.g., 0, 1, etc.)

[MASK_TYPE]:'MNAR', 'MAR', 'MCAR'


```
python main.py --dataname [NAME_OF_DATASET] --split_idx [MASK_IDX] --mask [MASK_TYPE]
```
Replace [DATASET_NAME], [MASK_IDX], and [MASK_TYPE] with your chosen values.

## Acknowledgement

We are deeply grateful for the valuable code and efforts contributed by the following GitHub repositories. Their contributions have been immensely beneficial to our work.
- https://github.com/state-spaces/mamba
- https://github.com/vanderschaarlab/hyperimpute
- https://github.com/hengruizhang98/DiffPuter

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@article{refidiff,
  title={RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation},
  volume={40},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/39034},
  DOI={10.1609/aaai.v40i24.39034},
  number={24},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Ahamed, Md Atik and Ye, Qiang and Cheng, Qiang},
  year={2026},
  month={Mar.},
  pages={19551-19559}
}
```

Thank you for using RefiDiff.
