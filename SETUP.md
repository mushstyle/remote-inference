# Setup

## Prerequisites

- Linux with GPU support
- Python 3.11

## Installation

```bash
apt-get install -y vim libgl1-mesa-glx
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## Clone and Install

```bash
mkdir -p ~/pkg
cd ~/pkg
git clone git@github.com:mushstyle/remote-inference.git
cd remote-inference
uv venv --python=3.11
source .venv/bin/activate
uv pip install -e .
```

## Configure

Set the API key (get the key from your infrastructure team or secrets manager):

```bash
./set-api-key.sh <YOUR_API_KEY>
```

## Run

```bash
remote-inference
```

The service listens on port 8000 by default.

## Health Check

```bash
curl http://localhost:8000/health
```
