#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = optimal_training_subset
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make virtual environment
.PHONY: make_venv
make_venv:
	$(PYTHON_INTERPRETER) -m venv .venv
	source .venv/bin/activate


## Install Python Dependencies
.PHONY: requirements
requirements:

	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 optimal_training_subset
	isort --check --diff --profile black optimal_training_subset
	black --check --config pyproject.toml optimal_training_subset

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml optimal_training_subset


## Run experiments
.PHONY: run_experiments
run_experiments:
	$(PYTHON_INTERPRETER) optimal_training_subset/experiments/FM_experiment.py
	$(PYTHON_INTERPRETER) optimal_training_subset/experiments/Cifar10-experiments



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
