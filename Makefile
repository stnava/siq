.PHONY: install clean test docs

install:
	pip install -e .

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage
	rm -rf .pytest_cache/
	rm -rf test_out*
	rm -f dummy_*.nii.gz

test:
	pytest tests/unit/ --cov=siq --cov-report=term-missing

docs:
	@echo "Documentation is located at docs/index.html"
	@echo "Opening docs in default browser..."
	python3 -c "import os, webbrowser; webbrowser.open('file://' + os.path.realpath('docs/index.html'))"
