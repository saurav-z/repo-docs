# RDFLib: Your Python Toolkit for Working with RDF Data

**RDFLib is a powerful and versatile Python library for parsing, manipulating, and serializing RDF (Resource Description Framework) data, enabling developers to build semantic web applications and work with linked data.** ([View on GitHub](https://github.com/RDFLib/rdflib))

## Key Features:

*   **Comprehensive RDF Support:** Parsers and serializers for RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, JSON-LD, and more.
*   **Flexible Graph Interface:**  Work with RDF data using a Pythonic `Graph` interface, backed by various store implementations.
*   **Multiple Store Implementations:**  Includes in-memory storage, persistent disk-based storage (Berkeley DB), and remote SPARQL endpoint support.
*   **SPARQL 1.1 Compliance:**  Full support for SPARQL 1.1 queries and update statements.
*   **Extensible Architecture:**  Plugin support for custom store implementations and SPARQL function extensions.

## What is RDF?

RDF (Resource Description Framework) is a standard model for data interchange on the Web. It enables you to represent information as triples of subject, predicate, and object, allowing for the creation of knowledge graphs and the linking of data across different sources.

## Getting Started

Install the stable release:

```bash
pip install rdflib
```

or with extras:
```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

Install the current main branch for developers:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

or

```bash
pip install -e .
```

## Examples

Here's how to create a graph, load RDF data, and print the results:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```
(See the original repo for more examples.)

## Other RDFlib Packages

The RDFLib community maintains several other RDF-related Python packages. These packages include:

*   [rdflib](https://github.com/RDFLib/rdflib) - The RDFLib core
*   [sparqlwrapper](https://github.com/RDFLib/sparqlwrapper) - A simple Python wrapper around a SPARQL service to remotely execute your queries
*   [pyLODE](https://github.com/RDFLib/pyLODE) - An OWL ontology documentation tool using Python and templating, based on LODE.
*   [pyrdfa3](https://github.com/RDFLib/pyrdfa3) - RDFa 1.1 distiller/parser library: can extract RDFa 1.1/1.0 from (X)HTML, SVG, or XML in general.
*   [pymicrodata](https://github.com/RDFLib/pymicrodata) - A module to extract RDF from an HTML5 page annotated with microdata.
*   [pySHACL](https://github.com/RDFLib/pySHACL) - A pure Python module which allows for the validation of RDF graphs against SHACL graphs.
*   [OWL-RL](https://github.com/RDFLib/OWL-RL) - A simple implementation of the OWL2 RL Profile which expands the graph with all possible triples that OWL RL defines.

Please see the list for all packages/repositories here:

*   <https://github.com/RDFLib>

## Documentation

Detailed documentation is available at: <https://rdflib.readthedocs.io>

## Contributing

We welcome contributions!  Please see the [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) for details.

*   **Pull Requests:** <https://github.com/RDFLib/rdflib/pulls>
*   **Issues:** <https://github.com/RDFLib/rdflib/issues>

## Support and Contact

*   **Stack Overflow:** Use the tag `rdflib` for "how do I..." questions: <https://stackoverflow.com/questions/tagged/rdflib>
*   **Mailing List:** rdflib-dev: <https://groups.google.com/group/rdflib-dev>
*   **Chat:** [Gitter](https://gitter.im/RDFLib/rdflib) or via matrix [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)