<h1 align="center">
    Unveiling Explainable Representation Learning in Earnings Call Transcripts via Structure-Aware Key Insight Extraction
</h1>

<br />

## Overview
This repo includes pytorch implementation of **EARNIE**. EARINE can automatically extract relevant information from earnings call transcripts. Our model leverages the structural information in transcripts to extract key insights effectively while providing concise explanations for each decision made by the model. We hope our research can shed light on the development of more efficient and effective transcript representation learning models for financial analysis. 

## Usage
Download and install the environment from the requirments file.
```
 /yourpath/anaconda3/envs/env_name/bin/python3.8 -m pip install -r requirements.txt 

conda activate env_name
```

See main.py for possible arguments.

use EARNIE to generate results to explain the key insights selection for earnings call transcripts:
```
python main.py --test True 
```

## Credit
[DeepVIB Repo](https://github.com/1Konny/VIB-pytorch): pytorch implementation of deep variational information bottleneck.

## References


## Contact

