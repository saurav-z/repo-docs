# RDFLib: A Powerful Python Library for Working with RDF Data

**RDFLib is your go-to Python library for parsing, manipulating, and serializing Resource Description Framework (RDF) data, enabling you to work with semantic web technologies.** Learn more at the [original repository](https://github.com/RDFLib/rdflib).

[![Build Status](https://github.com/RDFLib/rdflib/actions/workflows/validate.yaml/badge.svg?branch=main)](https://github.com/RDFLib/rdflib/actions?query=branch%3Amain)
[![Documentation Status](https://readthedocs.org/projects/rdflib/badge/?version=latest)](https://rdflib.readthedocs.io/en/latest/?badge=latest)
[![Coveralls branch](https://img.shields.io/coveralls/RDFLib/rdflib/main.svg)](https://coveralls.io/r/RDFLib/rdflib?branch=main)
[![GitHub stars](https://img.shields.io/github/stars/RDFLib/rdflib.svg)](https://github.com/RDFLib/rdflib/stargazers)
[![Downloads](https://pepy.tech/badge/rdflib/week)](https://pepy.tech/project/rdflib)
[![PyPI](https://img.shields.io/pypi/v/rdflib.svg)](https://pypi.python.org/pypi/rdflib)
[![PyPI](https://img.shields.io/pypi/pyversions/rdflib.svg)](https://pypi.python.org/pypi/rdflib)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6845245.svg)](https://doi.org/10.5281/zenodo.6845245)
[![Contribute with Gitpod](https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Gitter](https://badges.gitter.im/RDFLib/rdflib.svg)](https://gitter.im/RDFLib/rdflib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Matrix](https://img.shields.io/matrix/rdflib:matrix.org?label=matrix.org%20chat)](https://matrix.to/#/#RDFLib_rdflib:gitter.im)

## Key Features

*   **Comprehensive RDF Support:** Parse and serialize RDF data in various formats, including RDF/XML, N3, NTriples, Turtle, JSON-LD, and more.
*   **Flexible Graph Interface:** Work with RDF data using a powerful Graph interface, supporting multiple Store implementations for in-memory, persistent, and remote storage.
*   **SPARQL 1.1 Implementation:** Utilize a built-in SPARQL 1.1 implementation to query and update RDF data.
*   **Extensible Architecture:**  Extend functionality with SPARQL function extensions and custom store implementations via plugins.
*   **Mature and Active Community:** Benefit from a well-maintained library with a supportive community.

## RDFLib Family of Packages

The RDFLib community also maintains a family of related packages, including:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A wrapper for SPARQL services.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   and more!

## Versions and Releases

*   **Main Branch:** The current unstable release (version 8 alpha).
*   **Stable Releases:** See the releases page for details, including the latest stable release, and versioned builds.

    *   [Releases](https://github.com/RDFLib/rdflib/releases)

## Documentation

Comprehensive documentation is available at:

*   [https://rdflib.readthedocs.io](https://rdflib.readthedocs.io)

## Installation

Install the latest stable release using pip:

```bash
pip install rdflib
```

Install optional dependencies with extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

You can also install from the Git repository:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

or

```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```

## Getting Started

RDFLib provides a Pythonic API for working with RDF data. The primary data object is a `Graph`.

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD, FOAF

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)

#Example using namespaces
type = g.value(semweb, RDFS.label)

g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))

g.bind("foaf", FOAF)
g.bind("xsd", XSD)
print(g.serialize(format="turtle"))

# Adding a new namespace
dbpedia = Namespace('http://dbpedia.org/ontology/')
abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

## Running Tests

Tests are run with `pytest`.

```bash
poetry install
poetry run pytest
```

To generate a coverage report:

```bash
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcome!  See the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) to get started.

*   [Pull Requests](https://github.com/RDFLib/rdflib/pulls)
*   [Issues](https://github.com/RDFLib/rdflib/issues)
*   Gitpod and Google Cloud Shell options are provided.

## Support & Contacts

*   **Stack Overflow:** Use the tag `rdflib` for "how do I..." questions: <https://stackoverflow.com/questions/tagged/rdflib>
*   **Mailing List:**  rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   **Chat:** [Gitter](https://gitter.im/RDFLib/rdflib) or via matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)