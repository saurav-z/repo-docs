# wrapt: Powerful Python Decorators, Wrappers & Monkey Patching

**Enhance your Python code with robust and versatile decorators and wrappers using the `wrapt` module.**

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features

*   **Universal Decorators:** Works seamlessly with functions, methods (including classmethods and staticmethods), and classes.
*   **Transparent Object Proxies:** Provides advanced wrapping capabilities for sophisticated scenarios.
*   **Safe Monkey Patching Utilities:**  Modify Python code at runtime with confidence.
*   **Optimized Performance:**  Utilizes a C extension for speed, with a pure Python fallback.
*   **Introspection Preservation:**  Maintains function signatures, annotations, and other critical introspection data.
*   **Thread-Safe Implementations:** Ensures your decorators are safe to use in multi-threaded environments.

## Overview

The `wrapt` module provides a comprehensive solution for creating decorators, wrappers, and facilitating monkey patching in Python.  Built for correctness and flexibility, `wrapt` goes beyond standard decorator implementations to ensure compatibility across a wide range of Python use cases. Its design prioritizes preserving introspectability, function signatures, and type checking, making your decorators more reliable and predictable.

## Installation

```bash
pip install wrapt
```

## Quick Start

### Basic Decorator

```python
import wrapt

@wrapt.decorator
def pass_through(wrapped, instance, args, kwargs):
    return wrapped(*args, **kwargs)

@pass_through
def function():
    pass
```

### Decorator with Arguments

```python
import wrapt

def with_arguments(myarg1, myarg2):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        print(f"Arguments: {myarg1}, {myarg2}")
        return wrapped(*args, **kwargs)
    return wrapper

@with_arguments(1, 2)
def function():
    pass
```

### Universal Decorator

```python
import inspect
import wrapt

@wrapt.decorator
def universal(wrapped, instance, args, kwargs):
    if instance is None:
        if inspect.isclass(wrapped):
            # Decorator was applied to a class
            print("Decorating a class")
        else:
            # Decorator was applied to a function or staticmethod
            print("Decorating a function")
    else:
        if inspect.isclass(instance):
            # Decorator was applied to a classmethod
            print("Decorating a classmethod")
        else:
            # Decorator was applied to an instancemethod
            print("Decorating an instance method")
    
    return wrapped(*args, **kwargs)
```

## Documentation

Detailed documentation, examples, and advanced usage patterns are available at:

**[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss changes, report bugs, or ask questions.

**Note:** `wrapt` is a mature project.  Focus is on maintaining compatibility with current Python versions.

### Testing

See [TESTING.md](TESTING.md) for information on running tests.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation**: https://wrapt.readthedocs.io/
*   **PyPI**: https://pypi.python.org/pypi/wrapt
*   **Issues**: https://github.com/GrahamDumpleton/wrapt/issues/
*   **Changelog**: https://wrapt.readthedocs.io/en/latest/changes.html
*   **GitHub Repository**: [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)

## Related Blog Posts

*   [How you implemented your Python decorator is wrong](blog/01-how-you-implemented-your-python-decorator-is-wrong.md)
*   [The interaction between decorators and descriptors](blog/02-the-interaction-between-decorators-and-descriptors.md)
*   [Implementing a factory for creating decorators](blog/03-implementing-a-factory-for-creating-decorators.md)
*   [Implementing a universal decorator](blog/04-implementing-a-universal-decorator.md)
*   [Decorators which accept arguments](blog/05-decorators-which-accept-arguments.md)
*   [Maintaining decorator state using a class](blog/06-maintaining-decorator-state-using-a-class.md)
*   [The missing synchronized decorator](blog/07-the-missing-synchronized-decorator.md)
*   [The synchronized decorator as context manager](blog/08-the-synchronized-decorator-as-context-manager.md)
*   [Performance overhead of using decorators](blog/09-performance-overhead-of-using-decorators.md)
*   [Performance overhead when applying decorators to methods](blog/10-performance-overhead-when-applying-decorators-to-methods.md)
*   [Safely applying monkey patches in Python](blog/11-safely-applying-monkey-patches-in-python.md)
*   [Using wrapt to support testing of software](blog/12-using-wrapt-to-support-testing-of-software.md)
*   [Ordering issues when monkey patching in Python](blog/13-ordering-issues-when-monkey-patching-in-python.md)
*   [Automatic patching of Python applications](blog/14-automatic-patching-of-python-applications.md)
```

Key improvements:

*   **SEO Optimization:**  Added keywords like "Python Decorators", "Wrappers", and "Monkey Patching" in headings and descriptions.
*   **Hook:** Added a compelling one-sentence introduction.
*   **Structure:** Organized with clear headings and bullet points for readability.
*   **Conciseness:** Condensed information while retaining key details.
*   **Emphasis:** Highlighted key features for quick understanding.
*   **Links:** Included a direct link back to the GitHub repository.