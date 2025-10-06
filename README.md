# ProteinAE: Diffusion Protein AutoEncoder for Structure Encoding


<div align="center">
    <img width="600" alt="teaser" src="assets/overview.png"/>
</div>

<br>
<br>

## Setup
```
 
mamba activate proteinae
pip install -e .
```

## AutoEncoder

### Inference

```bash
python proteinfoundation/autoencode.py \
    --input_pdb $input_pdb \
    --output_dir output \
    --config_path /path/to/configs \
    --mode autoencode
```

### Training

```bash
python proteinfoundation/train_ae.py \
    --config_name training_ae_r1_d8
```


## LDM

### Training

```bash
python proteinfoundation/train_ldm.py \
    --config_name training_pldm_200M_afdb_512
```