# Serene Benchmark

The Serene Benchmark project provides a common framework to evaluate different approaches for schema matching and mapping.
Currently, the framework supports evaluation of three approaches for semantic typing of relational data sources.


### Prerequisites


To run evaluation benchmark,

1. The server for [Karma DSL](https://github.com/NICTA/iswc-2016-semantic-labeling) needs to be installed and started.
2. The server for [Serene](https://github.com/NICTA/serene) needs to be started.
3. Neural nets: tensorflow and keras python packages.
4. [Serene Python client](https://github.com/NICTA/serene-python-client) should be installed.



### How to test
Nose unittests needs to be installed. To run the tests:
```
nosetests
```

### Run

There are three different approaches for semantic typing which can be currently evaluated in this project:

1. DSL (domain-independent semantic labeller)
2. DINT (relational data integrator)
3. NNet (deep neural nets)

For NNetModel, allowed model types are: 'cnn@charseq' (CNN on character sequences), 'mlp@charfreq' (MLP on character freqs + entropy), 'rf@charfreq' (RF on character freqs + entropy)

DINT feature configuration is explained [here]{https://github.com/NICTA/serene/blob/master/matcher/dirstruct/semantic_type_classifier/repo/docs/features.txt},
and resampling strategy [here]{https://github.com/NICTA/serene/blob/master/matcher/dirstruct/semantic_type_classifier/HOWTO}.

DSL has only one default model with no further configuration available to the user.


To install the package 'serene-benchmark', run
```
python setup.py install
```

Running example...
