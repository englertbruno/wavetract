# Controlling a Vocal Tract With a Neural Network

## Installation 
1. **Environment setup.**
     ```bash 
    conda create -n wavetract python=3.10 && conda activate wavetract
    ```

2. **Install required packages.**
    ```bash
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1
    python3 -m pip install -r requirements.txt
    ```


## Data preparation
You can download the dataset from here: [ears dataset](https://sp-uhh.github.io/ears_dataset/).
Extract the zipfile and place the "ears" folder into the "out" folder under the project files.

## Usage

### Training
```bash
python3 main.py
```

## Citation
```
@article{englert2019wavetract,
  title   = "WaveTract: A hybrid generative model for speech synthesis",
  author  = "{Englert, Brunó B.} and {Zainkó, Csaba} and {Németh, Géza}",
  booktitle = "International Conference on Speech Technology and Human-Computer Dialogue (SpeD)",
  year    = "2019",
}
```
