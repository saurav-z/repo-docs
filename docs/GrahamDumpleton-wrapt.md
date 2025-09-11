# wrapt: Advanced Python Decorators, Wrappers, and Monkey Patching

**Unlock the power of Python with wrapt, a robust module designed to create sophisticated decorators, wrappers, and safe monkey patching solutions.** ([View on GitHub](https://github.com/GrahamDumpleton/wrapt))

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of wrapt

*   **Universal Decorators:** Work seamlessly with functions, methods, classmethods, staticmethods, and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping techniques for complex scenarios.
*   **Safe Monkey Patching:**  Provides utilities for making runtime modifications safely.
*   **Optimized Performance:** Includes a C extension for performance-critical components with a Python fallback.
*   **Introspection Preservation:** Maintains signatures, annotations, and other critical metadata.
*   **Thread-Safe Implementations:** Ensures reliability in multithreaded environments.

## Overview

The `wrapt` module is a powerful Python library providing the foundation for building function wrappers and decorators. It prioritizes correctness and addresses limitations in standard decorator implementations like `functools.wraps()`.  With `wrapt`, decorators preserve introspectability, signatures, and type-checking abilities, leading to more predictable and consistent behavior.

## Installation

Install `wrapt` easily using pip:

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

*   **Documentation**: [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Find the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to suggest changes, report bugs, or ask questions.

*Note:* `wrapt` is a mature project. The primary focus is on maintaining compatibility with evolving Python versions.

### Testing

Refer to [TESTING.md](TESTING.md) for information on running tests and available test commands.

## License

This project is licensed under the BSD License.  See the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation**: https://wrapt.readthedocs.io/
*   **PyPI**: https://pypi.python.org/pypi/wrapt
*   **Issues**: https://github.com/GrahamDumpleton/wrapt/issues/
*   **Changelog**: https://wrapt.readthedocs.io/en/latest/changes.html

## Related Blog Posts

The repository contains a series of blog posts explaining the design and implementation of wrapt:

-   [How you implemented your Python decorator is wrong](blog/01-how-you-implemented-your-python-decorator-is-wrong.md)
-   [The interaction between decorators and descriptors](blog/02-the-interaction-between-decorators-and-descriptors.md)
-   [Implementing a factory for creating decorators](blog/03-implementing-a-factory-for-creating-decorators.md)
-   [Implementing a universal decorator](blog/04-implementing-a-universal-decorator.md)
-   [Decorators which accept arguments](blog/05-decorators-which-accept-arguments.md)
-   [Maintaining decorator state using a class](blog/06-maintaining-decorator-state-using-a-class.md)
-   [The missing synchronized decorator](blog/07-the-missing-synchronized-decorator.md)
-   [The synchronized decorator as context manager](blog/08-the-synchronized-decorator-as-context-manager.md)
-   [Performance overhead of using decorators](blog/09-performance-overhead-of-using-decorators.md)
-   [Performance overhead when applying decorators to methods](blog/10-performance-overhead-when-applying-decorators-to-methods.md)
-   [Safely applying monkey patches in Python](blog/11-safely-applying-monkey-patches-in-python.md)
-   [Using wrapt to support testing of software](blog/12-using-wrapt-to-support-testing-of-software.md)
-   [Ordering issues when monkey patching in Python](blog/13-ordering-issues-when-monkey-patching-in-python.md)
-   [Automatic patching of Python applications](blog/14-automatic-patching-of-python-applications.md)