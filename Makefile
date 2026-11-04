PY ?= python

.PHONY: install data train evaluate all test api dashboard docker-build docker-pipeline docker-up

install:        ## install dependencies
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

data:           ## generate the synthetic dataset (Parquet + CSV)
	$(PY) -m aquatwin generate

train:          ## train detection, localisation (GCN/SAGE/GAT) and failure models
	$(PY) -m aquatwin train

evaluate:       ## realism levels 0-5 evaluation
	$(PY) -m aquatwin evaluate --episodes 3

all:            ## data + train + evaluate
	$(PY) -m aquatwin all --episodes 3

test:           ## physical-truth and engine tests
	$(PY) -m pytest -q

api:
	uvicorn aquatwin.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

docker-build:
	docker compose build

docker-pipeline:
	docker compose --profile batch run --rm pipeline

docker-up:
	docker compose up api dashboard
