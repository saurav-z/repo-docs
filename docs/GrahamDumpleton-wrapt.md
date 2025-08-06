# wrapt: Python Decorators, Wrappers, and Monkey Patching - Done Right

**wrapt** is a powerful Python module designed to create robust and reliable decorators, wrappers, and facilitate safe monkey patching, ensuring your code remains introspectable and performs optimally.

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features

*   **Universal Decorators:** Works seamlessly with functions, methods (instance, class, static), and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping techniques for complex scenarios.
*   **Safe Monkey Patching Utilities:** Provides tools for runtime code modification with enhanced safety.
*   **Optimized Performance:** Leverages a C extension for speed, with a Python fallback for wider compatibility.
*   **Comprehensive Introspection Preservation:** Preserves signatures, annotations, and other critical metadata.
*   **Thread-Safe Implementations:** Guarantees safe operation in multi-threaded environments.

## Overview

The **wrapt** module is a cornerstone for creating reliable and versatile decorators and wrappers in Python. It goes beyond the capabilities of `functools.wraps()` by meticulously preserving introspectability, signatures, and type-checking abilities.  This results in decorators that function correctly in a wider range of situations and provide consistent, predictable behavior.  Performance is prioritized with a C extension module for speed-critical operations, alongside a pure Python fallback for systems without a compiler.

## Installation

Install **wrapt** easily using pip:

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

For in-depth information, examples, and advanced usage, explore the comprehensive documentation:

**[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to suggest changes, report issues, or discuss how things work.

Note: wrapt is now a mature project, with a primary focus on maintaining compatibility with newer Python versions and the evolving ecosystem.

### Testing

See [TESTING.md](TESTING.md) for test details, including Python version-specific conventions and test commands.

## License

This project is licensed under the BSD License; see the [LICENSE](LICENSE) file.

## Links

*   **Documentation**: [https://wrapt.readthedocs.io/](https://wrapt.readthedocs.io/)
*   **PyPI**: [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues**: [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog**: [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)
*   **Source Code:** [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)

## Related Blog Posts

Explore the design and implementation of wrapt through a series of blog posts:

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