# Making Pandas better

This repository was forked from Pandas master branch. 

It contains several improvements to the original Pandas API:

- DataFrame
  - builder
  - indexing
  - mean
- DatetimeIndex
  - builder

### How to build

1. Clone this repo.
2. [Optional] Create and open a virtual environment for dependencies:
   `virtualenv venv`
   `source venv/bin/activate`
3. Install dependencies: 
   `pip install -r requirements-dev.txt` 

3. Build numpy extensions into C:
   `python setup.py build_ext --inplace --force`

### How to run demo

TBD

### How to run tests

To run tests for the whole Pandas project, use:

`python -m pytest` 

Because we only touched a few files, here is a minimal set of related tests:

`python -m pytest pandas/tests/frame/test_api.py` 

`python -m pytest pandas/tests/indexes/datetimes/test_construction.py` 