# cryptolabeller

Prerequisites:
- [pipenv](https://pipenv.pypa.io/en/latest/)
- Assumes you have a mongodb running with OHLC data.

Entrypoint for training/preprocessing `src/main.py`
Tensorboard logging outputted at `src/logs/finance`

Validation data has been persisted to the repo to analyze results:

To install python environment
```bash
pipenv install
```
Activate the pipenv environment
```bash
pipenv shell
```
To run streamlit app
```bash
cd src
streamlit run analyze.py
```