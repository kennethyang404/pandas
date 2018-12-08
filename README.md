# Making Pandas better

This repository was forked from Pandas master branch. 

### How to build

1. Clone this repo. Go into the directory.

2. [Optional] Create and open a virtual environment for dependencies:

   ```
   $ virtualenv venv
   $ source venv/bin/activate
   ```

3. Install dependencies: 

   ```
   $ pip install -r requirements-dev.txt
   ```

3. Build numpy extensions into C:

   ```
   $ python setup.py build_ext --inplace --force
   ```


### How to play with it

In the root directory, open python interpreter

```
$ python
```

```
> import pandas as pd
```

Have fun!

### How to run tests

Because we only touched a few files, here is a minimal set of related tests:

```
$ python -m pytest pandas/tests/frame/test_creation.py
$ python -m pytest pandas/tests/frame/test_api.py
$ python -m pytest pandas/tests/indexes/datetimes/test_construction.py
```

### Where are our changes

Besides the tests, here are the major files we touched:

For DataFrame, we mainly modified pandas/core/frame.py file. We added a nested class Builder to DataFrame and four class methods to produce Builders. They can be found toward to end of DataFrame class definition.

For DatetimeIndex, we also added a Builder class in file pandas/core/indexes/datetimes.py and associated class methods to produce the builder. 

If you find it difficult to find them, see recent commits for details. 