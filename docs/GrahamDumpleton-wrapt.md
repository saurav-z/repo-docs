# wrapt: Python Decorators, Wrappers, and Monkey Patching for Robust Code

**Enhance your Python code with `wrapt`, a powerful module for creating decorators, wrappers, and monkey patching that prioritizes correctness and performance.**  [See the original repository](https://github.com/GrahamDumpleton/wrapt)

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of wrapt

`wrapt` offers a comprehensive set of features designed to make your code more reliable and easier to maintain:

*   **Universal Decorators:** Works seamlessly with functions, methods (including instance, class, and static), and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping techniques.
*   **Robust Monkey Patching:** Provides utilities for safe runtime modifications, reducing the risk of unexpected behavior.
*   **Optimized Performance:** Utilizes a C extension module for speed-critical components, with a Python fallback for broader compatibility.
*   **Preserves Introspection:** Ensures decorators maintain function signatures, annotations, and other critical metadata.
*   **Thread-Safe Implementations:**  Offers built-in thread safety for reliable concurrent execution.

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

Comprehensive documentation, examples, and usage patterns are available on [wrapt.readthedocs.io](https://wrapt.readthedocs.io/).

## Supported Python Versions

`wrapt` supports:

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome! Please submit issues or pull requests via the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/). Note that `wrapt` is a mature project, so major feature additions are unlikely, but improvements, bug fixes, and compatibility updates are always appreciated.

## Testing

For information on running tests, refer to [TESTING.md](TESTING.md).

## License

`wrapt` is licensed under the BSD License. See the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation:** [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **PyPI:** [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues:** [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)

## Related Blog Posts

Explore the design and implementation of `wrapt` through a series of blog posts:

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