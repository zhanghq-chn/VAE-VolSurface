# Volatility Surface fitting with Variational Autoencoders

## Configuration

### Install dependencies

```shell
conda env create -f environment.yml
conda activate vae-volsurface
```
or creating a virtual environment
```shell
python -m venv vae-volsurface
source vae-volsurface/bin/activate
pip install -r requirements.txt
deactivate # to leave the virtual environment
```


### Setup data folder
1. Copy `.env_sample` as `.env`
2. Add your own data directory to `.env`

### Linting

Run `nox` before commiting.

## Reference

- Variational Autoencoders: A Hands-Off Approach to Volatility
