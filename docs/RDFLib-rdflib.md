# RDFLib: Your Go-To Python Library for Working with RDF

**RDFLib is a powerful and versatile Python library for parsing, manipulating, and serializing Resource Description Framework (RDF) data, enabling you to work with semantic web technologies seamlessly.** [View the original repository](https://github.com/RDFLib/rdflib)

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

*   **Comprehensive RDF Support:** Parses and serializes RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and HexTuples.
*   **Flexible Graph Interface:**  Provides a Graph interface that can be backed by various store implementations.
*   **Diverse Store Implementations:** Includes in-memory, persistent on-disk (Berkeley DB), and remote SPARQL endpoint store options, with plugin support for additional stores.
*   **SPARQL 1.1 Implementation:** Supports SPARQL 1.1 queries and update statements.
*   **SPARQL Function Extensions:** Provides mechanisms to extend SPARQL functionality.
*   **Pythonic API:** Designed to be user-friendly and intuitive for Python developers.

## RDFLib Family of Packages

The RDFLib community maintains a suite of related Python packages to support diverse RDF-related tasks:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A simple Python wrapper for SPARQL services.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa 1.1 distiller/parser library.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - A module to extract RDF from microdata in HTML5.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - A module for validating RDF graphs against SHACL graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL2 RL Profile implementation.

Explore the full list of RDFLib packages: <https://github.com/RDFLib>

## Versions & Releases

* `main` branch in this repository is the current unstable release - version 8 alpha
* `7.1.4` tidy-up release, possibly last 7.x release
* `7.1.3` current stable release, small improvements to 7.1.1
* `7.1.2` previously deleted release
* `7.1.1` previous stable release
    * see <https://github.com/RDFLib/rdflib/releases/tag/7.1.1>
* `7.0.0` previous stable release, supports Python 3.8.1+ only.
    * see [Releases](https://github.com/RDFLib/rdflib/releases)
* `6.x.y` supports Python 3.7+ only. Many improvements over 5.0.0
    * see <https://github.com/RDFLib/rdflib/releases/tag/6.3.2>
* `5.x.y` supports Python 2.7 and 3.4+ and is [mostly backwards compatible with 4.2.2](https://rdflib.readthedocs.io/en/stable/upgrade4to5.html).
  * * see <https://github.com/RDFLib/rdflib/releases/tag/5.0.0>

See <https://github.com/RDFLib/rdflib/releases/> for the release details.

## Documentation

Comprehensive documentation is available at <https://rdflib.readthedocs.io>, including `latest`, `stable`, and versioned builds.

## Installation

Install the stable release of RDFLib using pip:

```bash
pip install rdflib
```

Install optional dependencies with pip extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download the package from PyPI: <https://pypi.python.org/pypi/rdflib>

### Installing the Current Main Branch (for Developers)

Install from the Git repository using pip:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

or

```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```

Or install from your local clone:

```bash
poetry install  # installs into a poetry-managed venv
```

or

```bash
pip install -e .
```

## Getting Started

RDFLib uses a `Graph` object to store RDF triples (Subject, Predicate, Object):

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

Components of the triples are URIs (resources) or Literals (values).

Use common namespaces like so:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```
Where `RDFS` is the RDFS namespace, `XSD` the XML Schema Datatypes namespace and `g.value` returns an object of the triple-pattern given (or an arbitrary one if multiple exist).

Adding triples:

```python
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

Serializing to Turtle:

```python
g.bind("foaf", FOAF)
g.bind("xsd", XSD)

print(g.serialize(format="turtle"))
```

Output:

```turtle
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

<http://example.com/person/nick> foaf:givenName "Nick"^^xsd:string .
```

Define custom namespaces:

```python
dbpedia = Namespace('http://dbpedia.org/ontology/')

abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

See [./examples](./examples) for more.

## Running Tests

### Running the tests on the host

Run the test suite with `pytest`.
```shell
poetry install
poetry run pytest
```

### Running test coverage on the host with coverage report

Run the test suite and generate a HTML coverage report with `pytest` and `pytest-cov`.
```shell
poetry run pytest --cov
```

### Viewing test coverage

Once tests have produced HTML output of the coverage report, view it by running:
```shell
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contribute to RDFLib!  See the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) to get started. Submit pull requests:

*   <https://github.com/RDFLib/rdflib/pulls>

Consider using Gitpod or Google Cloud Shell for development:

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Report issues:

*   <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

For general questions, use Stack Overflow with the tag `rdflib`:

*   <https://stackoverflow.com/questions/tagged/rdflib>

Contact maintainers via:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Chat on [Gitter](https://gitter.im/RDFLib/rdflib) or [Matrix](https://matrix.to/#/#RDFLib_rdflib:gitter.im)