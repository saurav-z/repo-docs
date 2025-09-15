# RDFLib: Your Python Library for Working with RDF Data

**RDFLib is a powerful and versatile Python library for parsing, manipulating, and serializing RDF (Resource Description Framework) data.**

[Go to the RDFLib Repository](https://github.com/RDFLib/rdflib)

**Key Features:**

*   **Comprehensive RDF Support:** Parse and serialize RDF data in various formats, including RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig, JSON-LD, and HexTuples.
*   **Flexible Graph Interface:** Utilize a Graph interface with diverse Store implementations, supporting in-memory, persistent (Berkeley DB), and remote SPARQL endpoint storage options.  Additional stores can be added via plugins.
*   **SPARQL 1.1 Implementation:**  Execute SPARQL 1.1 queries and update statements directly within your Python code.
*   **Extensible Architecture:** Benefit from SPARQL function extension mechanisms for customized functionality.
*   **Broad Format Support:**  Includes parsers for RDFa and Microdata extraction.

## Installation

Install the stable release using pip:

```bash
pip install rdflib
```

Install optional dependencies with extras:

```bash
pip install rdflib[berkeleydb,networkx,html,lxml,orjson]
```

For developers, install the current main branch from the Git repository:

```bash
pip install git+https://github.com/rdflib/rdflib@main
```

## Getting Started

RDFLib employs a Pythonic RDF API, with the `Graph` object as its primary data structure. Graphs store RDF triples (Subject, Predicate, Object).

Here's a simple example:

```python
from rdflib import Graph
g = Graph()
g.parse('http://dbpedia.org/resource/Semantic_Web')

for s, p, o in g:
    print(s, p, o)
```

You can use common namespaces from `rdflib.namespace`:

```python
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, XSD

g = Graph()
semweb = URIRef('http://dbpedia.org/resource/Semantic_Web')
type = g.value(semweb, RDFS.label)
```

## Documentation and Releases

*   [Documentation](https://rdflib.readthedocs.io/)
*   [Releases](https://github.com/RDFLib/rdflib/releases/)

## Contributing

We welcome contributions! Please refer to our [contributing guide](https://rdflib.readthedocs.io/en/latest/CONTRIBUTING/) and [developers guide](https://rdflib.readthedocs.io/en/latest/developers/) and submit pull requests.

## Support

*   **Stack Overflow:**  Use the tag `rdflib` for "how do I..." questions: <https://stackoverflow.com/questions/tagged/rdflib>
*   **Mailing List:** rdflib-dev: <https://groups.google.com/group/rdflib-dev>
*   **Chat:** Gitter: <https://gitter.im/RDFLib/rdflib> or Matrix: [#RDFLib_rdflib:gitter.im](https://matrix.to/#/#RDFLib_rdflib:gitter.im)