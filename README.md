# Serene Benchmark

The Serene Benchmark project provides a common framework to evaluate different approaches for schema matching and mapping.
Currently, the framework supports evaluation of three approaches for semantic typing of relational data sources.


### Prerequisites


To run evaluation benchmark,

1. The server for [Karma DSL](https://github.com/NICTA/iswc-2016-semantic-labeling) needs to be installed.
2. The server for [Serene](https://github.com/NICTA/serene) needs to be started.
3. Neural nets: tensorflow and keras python packages.
4. [Serene Python client](https://github.com/NICTA/serene-python-client) should be installed.



### How to test
Nose unittests needs to be installed. To run the tests:
```
nosetests
```

### Run

To install the package 'serene-benchmark', run
```
python setup.py install
```

Running example...
