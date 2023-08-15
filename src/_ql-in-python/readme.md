# build the 1st time
```sh
python -m venv .env
source .env/bin/activate
pip install maturin
maturin develop
```

# build from 2nd time on
```sh
source .env/bin/activate
maturin develop
```
