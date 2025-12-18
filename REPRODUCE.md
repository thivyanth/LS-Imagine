# Reproduction Guide

This guide outlines the steps to reproduce the environment and run the LS-Imagine pipeline.

## Prerequisites

- Conda (Miniconda or Anaconda)
- NVIDIA GPU with CUDA 12.1+ support
- Git LFS (`sudo apt-get install git-lfs`)

## Setup Instructions

### 1. Environment Setup

Run the provided script to create the conda environment and install dependencies:

```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
```

This will:
- Create a conda environment named `ls` (Python 3.9).
- Install PyTorch with CUDA 12.1 support.
- Install necessary build tools and Python packages from `requirements.txt`.
- Attempt to install `minedojo`.

### 2. JDK Installation

The repository includes JDK 1.8 tarballs in `external/jdk/`. To use them:

```bash
mkdir -p ~/java
tar -xzf external/jdk/jdk-8u202-linux-x64.tar.gz -C ~/java
export JAVA_HOME=~/java/jdk1.8.0_202
export PATH=$JAVA_HOME/bin:$PATH
```

Verify the installation:
```bash
java -version
```

### 3. Verify MineDojo Installation

Run the test script (ensure headless mode if on a server):

```bash
MINEDOJO_HEADLESS=1 python scripts/test_minedojo.py
```

### 4. Running the Pipeline

Use the `pipeline.sh` script to run the full LS-Imagine pipeline:

```bash
# Example: mine iron ore
bash ./scripts/pipeline.sh mine_iron_ore "find and mine iron ore"
```

Refer to `scripts/pipeline.sh` for more options and examples.

## Troubleshooting

For detailed troubleshooting, especially regarding MineDojo and Gradle errors, refer to [docs/minedojo_installation.md](docs/minedojo_installation.md).

