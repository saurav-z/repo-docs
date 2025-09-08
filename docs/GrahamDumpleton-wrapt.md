# wrapt: Robust Python Decorators, Wrappers, and Monkey Patching

**Enhance your Python code with wrapt, a powerful module for creating decorators, wrappers, and facilitating safe monkey patching for improved code maintainability and performance.  Learn more about wrapt at [github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)**

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (instance, class, and static), and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping and modification of objects.
*   **Safe Monkey Patching:**  Provides utilities for safely modifying code at runtime.
*   **Optimized Performance:**  Includes a C extension for speed-critical operations with a Python fallback.
*   **Introspection Preservation:**  Maintains function signatures, annotations, and other critical metadata.
*   **Thread-Safe Decorator Implementations:**  Ensures safe usage in multi-threaded environments.

## Installation

Install wrapt using pip:

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

Comprehensive documentation, examples, and advanced usage patterns can be found at:

*   **[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome! Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to suggest changes, report bugs, or ask questions.

**Note:** Wrapt is a mature project with a focus on maintaining compatibility and ensuring correct behavior with new Python versions. New feature additions are unlikely.

### Testing

See [TESTING.md](TESTING.md) for information on running tests.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation**: https://wrapt.readthedocs.io/
*   **PyPI**: https://pypi.python.org/pypi/wrapt
*   **Issues**: https://github.com/GrahamDumpleton/wrapt/issues/
*   **Changelog**: https://wrapt.readthedocs.io/en/latest/changes.html

## Related Blog Posts

The repository also includes a series of blog posts detailing the design and implementation of wrapt.  These posts provide in-depth insights into the workings of the library.