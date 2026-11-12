.PHONY: install test train train-aod train-physdehaze eval benchmark figures demo app tensorboard verify-data clean help

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
STREAMLIT ?= .venv/bin/streamlit
TENSORBOARD ?= .venv/bin/tensorboard

help:
	@echo "Available commands:"
	@echo "  make install            Install dependencies from requirements.txt"
	@echo "  make test               Run full test suite with pytest"
	@echo "  make train              Train PhysDehaze-Lite with default settings"
	@echo "  make train-aod          Train AOD-Net baseline"
	@echo "  make train-physdehaze   Train PhysDehaze-Lite proposed model"
	@echo "  make eval               Run evaluation on test split"
	@echo "  make benchmark          Run multi-method benchmark suite (DCP, AOD-Net, PhysDehaze-Lite)"
	@echo "  make figures            Generate publication-quality analysis figures (Uncertainty, Scatter, Visual Grid)"
	@echo "  make demo               Run single-image dehazing demo and save demo_result.png"
	@echo "  make app                Launch interactive Streamlit web application"
	@echo "  make tensorboard        Launch TensorBoard monitoring dashboard on port 6006"
	@echo "  make verify-data        Verify dataset pair matching and image integrity"
	@echo "  make clean              Remove cache files and temporary artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTEST) tests/ -v

verify-data:
	$(PYTHON) scripts/verify_dataset.py --data-dir data/Haze1k

train:
	$(PYTHON) src/train/train.py --model physdehaze_lite --epochs 10 --batch-size 8

train-aod:
	$(PYTHON) src/train/train.py --model aodnet --epochs 10 --batch-size 8

train-physdehaze:
	$(PYTHON) src/train/train.py --model physdehaze_lite --epochs 10 --batch-size 8

eval:
	$(PYTHON) scripts/evaluate_benchmarks.py --data-dir data/Haze1k --samples-per-subset 20

benchmark:
	$(PYTHON) scripts/run_benchmark.py --data-dir data/Haze1k --rice-dir data/RICE_DATASET/RICE1 --output-dir experiments/results

figures:
	$(PYTHON) scripts/generate_paper_figures.py --output-dir experiments/figures

demo:
	$(PYTHON) scripts/demo.py --output demo_result.png

app:
	$(STREAMLIT) run app/streamlit_app.py

tensorboard:
	$(TENSORBOARD) --logdir experiments/ --port 6006

clean:
	rm -rf __pycache__ .pytest_cache experiments/tmp*
	find . -type d -name "__pycache__" -exec rm -rf {} +

