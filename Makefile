flashinstall:
	cd flash-attention; python setup.py install

install:
	pip install -e .

run: install
	python -m analysis_node.app --config ./config.yaml --device cpu

update:
	conda env update -f environment.yaml --prune

create:
	conda env create -f environment.yaml