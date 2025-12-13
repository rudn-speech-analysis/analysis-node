# Speech analysis pipeline

## Preparing the environment

### 1. Download the code

Clone this git repo locally.

```bash
$ git clone https://github.com/rudn-speech-analysis/analysis-node
$ cd analysis-node
```

Update the submodules.

```bash
git submodule init && git submodule update --recursive
```

### 2. Create the conda environment

```bash
$ conda env create -f ./environment.yaml
```

Activate the environment:

```bash
$ conda activate speech-analysis
```

### 3. Install flash-attn

#### Nvidia

For Nvidia GPUs installing directly with `pip` or `conda` is enough

#### AMD

For AMD GPUs build flash-attn yourself.

`speech-analysis` environment already defines some `flash-attn`'s
environment variables, but you can set them yourself:

```bash
$ export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
$ export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE
```

Then build the library:

```bash
$ cd flash-attention
$ python setup.py install
```
