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

## SSH Access

The following public keys should be added to `~/.ssh/authorized_keys` on each GPU instance to allow SSH access from authorized machines:

```
# tim local
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE96qyv1tYzb0/CsmLPDJ46JK9qC3huXPptALFYVRIQQ blah@blahs-MacBook-Air.local

# will local
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIzOqRsrojm2IH1U4b47MQcDfg77FLUuQFZcjB9+Kkg/ williamgalebach@Williams-MacBook-Air-2022.local

# dev server
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILgHZV60fAwYkr1k0doJwfIxToVNwNUr1Pv2CBCfE+0H tim@development-server
```

To add these keys on a GPU instance:

```bash
mkdir -p ~/.ssh
cat >> ~/.ssh/authorized_keys << 'KEYS'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE96qyv1tYzb0/CsmLPDJ46JK9qC3huXPptALFYVRIQQ blah@blahs-MacBook-Air.local
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIzOqRsrojm2IH1U4b47MQcDfg77FLUuQFZcjB9+Kkg/ williamgalebach@Williams-MacBook-Air-2022.local
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILgHZV60fAwYkr1k0doJwfIxToVNwNUr1Pv2CBCfE+0H tim@development-server
KEYS
chmod 600 ~/.ssh/authorized_keys
```
