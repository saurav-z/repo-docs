# RDFLib: Your Go-To Python Library for Working with RDF Data

**RDFLib** is a powerful and versatile Python library that empowers developers to work with RDF (Resource Description Framework) data, enabling seamless semantic web applications. You can find the original repository [here](https://github.com/RDFLib/rdflib).

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

*   **Comprehensive RDF Support:** Parsers and serializers for RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and more.
*   **Flexible Graph Interface:**  A `Graph` interface that can be backed by various store implementations.
*   **Versatile Store Implementations:** Includes in-memory storage, persistent storage (Berkeley DB), and SPARQL endpoint integration.
*   **SPARQL 1.1 Implementation:** Supports SPARQL 1.1 queries and update statements for powerful data manipulation.
*   **Extensible:**  Supports SPARQL function extension mechanisms.

## RDFLib Family of Packages

The RDFLib community offers a suite of related Python packages:

*   [rdflib](https://github.com/RDFLib/rdflib) - The core RDFLib library.
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A SPARQL service wrapper.
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa 1.1 distiller/parser.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - Extracts RDF from HTML5 microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - Validates RDF graphs against SHACL graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - OWL2 RL Profile implementation.

For a complete list, see:  <https://github.com/RDFLib>

## Versions & Releases

*   `main` - Current unstable release - version 8 alpha
*   `7.2.1` - tiny clean up release, relaxes Python version requirement
*   `7.2.0` - general fixes and usability improvements
*   `7.1.4` - tidy-up release, possibly last 7.x release
*   `7.1.3` - current stable release
*   `7.1.1` - previous stable release
*   `7.0.0` - previous stable release, supports Python 3.8.1+ only.
*   `6.x.y` - supports Python 3.7+ only.
*   `5.x.y` - supports Python 2.7 and 3.4+ and is [mostly backwards compatible with 4.2.2](https://rdflib.readthedocs.io/en/stable/upgrade4to5.html).

See [Releases](https://github.com/RDFLib/rdflib/releases) for details.

## Documentation

Explore comprehensive documentation at <https://rdflib.readthedocs.io>

## Installation

Install the stable release using *pip*:

```bash
pip install rdflib
```

Install optional dependencies with extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Alternatively, download from PyPI: <https://pypi.python.org/pypi/rdflib>

### Installing the Current Main Branch (for Developers)

Install from the Git repository using *pip*:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

or

```bash
pip install -e git+https://github.com/rdflib/rdflib@main#egg=rdflib
```

or from your locally cloned repository:

```bash
poetry install  # installs into a poetry-managed venv
```

or

```bash
pip install -e .
```

## Getting Started

RDFLib uses a `Graph` object as its primary data structure, which is a collection of RDF *Subject, Predicate, Object* Triples:

Here's how to create a graph, load data from DBpedia, and print results:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```
Triples use URIs (resources) or Literals (values).
Common namespaces are included in RDFLib, e.g.:

```python
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
```

Example usage:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

Adding a triple:

```python
g.add((
    URIRef("http://example.com/person/nick"),
    FOAF.givenName,
    Literal("Nick", datatype=XSD.string)
))
```

Bind namespaces for concise serialization:

```python
g.bind("foaf", FOAF)
g.bind("xsd", XSD)
```

Then serialize:

```python
print(g.serialize(format="turtle"))
```

New Namespaces can also be defined:

```python
dbpedia = Namespace('http://dbpedia.org/ontology/')

abstracts = list(x for x in g.objects(semweb, dbpedia['abstract']) if x.language=='en')
```

See [./examples](./examples) for more examples.

## Features Summary

*   **RDF Format Support:** Parsers and serializers for a wide array of RDF formats.
*   **Graph Interface with Store Implementations:** Flexible storage options.
*   **SPARQL 1.1:**  Implementations for queries and updates.

## Running Tests

### Running Tests on the Host

Run the test suite with `pytest`:

```bash
poetry install
poetry run pytest
```

### Running Test Coverage on the Host

Run tests and generate an HTML coverage report:

```bash
poetry run pytest --cov
```

### Viewing Test Coverage

View the HTML coverage report:

```bash
poetry run pytest --cov --cov-report term --cov-report html
python -m http.server --directory=htmlcov
```

## Contributing

Contribute to RDFLib! Read the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) to get started.  Submit pull requests here:

*   <https://github.com/RDFLib/rdflib/pulls>

Use Gitpod or Google Cloud Shell for development:

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/RDFLib/rdflib)
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FRDFLib%2Frdflib&cloudshell_git_branch=main&cloudshell_open_in_editor=README.md)

Report issues:

*   <https://github.com/RDFLib/rdflib/issues>

## Support & Contacts

For general questions, use Stack Overflow with the `rdflib` tag:
*   <https://stackoverflow.com/questions/tagged/rdflib>

Contact the maintainers:

*   rdflib-dev mailing list: <https://groups.google.com/group/rdflib-dev>
*   Gitter chat: [gitter](https://gitter.im/RDFLib/rdflib) or via matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)