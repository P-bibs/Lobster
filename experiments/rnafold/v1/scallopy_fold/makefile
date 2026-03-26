develop:
	pip install --editable .

install:
	find dist -name "*.whl" -print | xargs pip install --force-reinstall

build:
	python -m build

clean:
	rm -rf dist
