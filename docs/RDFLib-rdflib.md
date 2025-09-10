<div align="center">
  <img src="docs/_static/RDFlib.png" alt="RDFLib Logo" width="200"/>
  <h1>RDFLib: Your Python Toolkit for Working with RDF Data</h1>
</div>

RDFLib is a powerful and versatile Python library that empowers developers to work with Resource Description Framework (RDF) data, the foundation of the Semantic Web. Find the original repo [here](https://github.com/RDFLib/rdflib).

<br/>

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
*   **Flexible Graph Interface:** Work with RDF data using a Graph interface, backed by multiple store implementations, including in-memory, persistent disk-based (Berkeley DB), and remote SPARQL endpoints.
*   **SPARQL 1.1 Implementation:**  Execute SPARQL 1.1 queries and update statements directly within your Python code.
*   **Extensible:**  Easily integrate custom SPARQL functions and extend RDFLib's capabilities.
*   **Mature and Active:**  Benefit from a well-established library with a vibrant community and ongoing development.

## What is RDFLib?

RDFLib is a pure Python library that provides a robust and Pythonic interface for working with RDF data. Whether you're building semantic web applications, knowledge graphs, or data integration solutions, RDFLib offers the tools you need to handle RDF effectively.

## RDFLib Family of Packages

The RDFLib community maintains a collection of related Python packages, including:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A SPARQL service wrapper.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa distiller/parser library.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - Microdata extraction from HTML5.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - SHACL validation in Python.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL2 RL Profile implementation.

Explore the full range of RDFLib projects at [https://github.com/RDFLib](https://github.com/RDFLib/).

## Versions & Releases

*   `main` branch: Current unstable release (version 8 alpha)
*   `7.1.4`: Tidy-up release, possibly last 7.x release
*   `7.1.3`: Current stable release with improvements.
*   `7.1.1`: Previous stable release.
*   `7.0.0`: Previous stable release, Python 3.8.1+ only.
*   `6.x.y`: Supports Python 3.7+ only, with many improvements over 5.0.0.
*   `5.x.y`: Supports Python 2.7 and 3.4+ and is [mostly backwards compatible with 4.2.2](https://rdflib.readthedocs.io/en/stable/upgrade4to5.html).

See <https://github.com/RDFLib/rdflib/releases/> for release details.

## Documentation

Comprehensive documentation is available at: <https://rdflib.readthedocs.io>

## Installation

Install the latest stable release using pip:

```bash
pip install rdflib
```

Install optional dependencies (e.g., for Berkeley DB, networkx, HTML parsing, etc.):

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

You can also install from the Git repository:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

## Getting Started

RDFLib provides a Pythonic API for working with RDF data.  The core object is the `Graph`, which stores RDF triples.

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD

# Create a new graph
g = Graph()

# Parse RDF data from a URL (e.g., DBpedia)
g.parse('http://dbpedia.org/resource/Semantic_Web')

# Iterate through triples and print
for s, p, o in g:
    print(s, p, o)

# Define a namespace
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)

# Add triples to the graph
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))

# Serialize the graph in Turtle format
g.bind("foaf", FOAF)
g.bind("xsd", XSD)
print(g.serialize(format="turtle"))

# Create a new namespace
dbpedia = Namespace('http://dbpedia.org/ontology/')
abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

## Features

*   Parsers and serializers for various RDF formats (RDF/XML, N3, NTriples, etc.).
*   Graph interface with multiple store implementations (in-memory, Berkeley DB, SPARQL endpoints).
*   SPARQL 1.1 query and update support.
*   Extendable through plugins.

## Running Tests

### Running the tests on the host

```shell
poetry install
poetry run pytest
```

### Running test coverage on the host with coverage report

```shell
poetry run pytest --cov
```

### Viewing test coverage

```shell
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contributions are welcome!  Refer to the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) to get started.

Submit Pull Requests: <https://github.com/RDFLib/rdflib/pulls>

Use Gitpod or Google Cloud Shell to set up a development environment.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

## Support & Contacts

*   **General Questions:** Use Stack Overflow and tag your questions with `rdflib`:  <https://stackoverflow.com/questions/tagged/rdflib>
*   **Mailing List:**  <https://groups.google.com/group/rdflib-dev>
*   **Chat:** Gitter:  <https://gitter.im/RDFLib/rdflib> or Matrix: [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)