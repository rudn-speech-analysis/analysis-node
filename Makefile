flashinstall:
	cd flash-attention; python setup.py install

install:
	pip install -e .

run: install
	python -m analysis_node.app --config ./config.yaml --device cuda
