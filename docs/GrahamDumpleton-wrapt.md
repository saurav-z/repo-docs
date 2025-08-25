# Wrapt: Powerful Python Decorators, Wrappers, and Monkey Patching

**Wrapt is a robust Python library designed to simplify and enhance the creation of decorators, wrappers, and monkey patching techniques.**

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of Wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (including classmethods and staticmethods), and classes.
*   **Transparent Object Proxies:** Provides advanced wrapping capabilities for complex scenarios.
*   **Safe Monkey Patching:** Includes utilities for safely modifying code at runtime.
*   **Optimized Performance:** Utilizes a C extension for speed, with a Python fallback for compatibility.
*   **Introspection Preservation:** Preserves function signatures, annotations, and other critical information.
*   **Thread-Safe Implementations:** Ensures decorators work correctly in multi-threaded environments.

## Installation

Install Wrapt using pip:

```bash
pip install wrapt
```

## Getting Started

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

Comprehensive documentation, examples, and usage guides are available at:

[https://wrapt.readthedocs.io/](https://wrapt.readthedocs.io/)

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss any changes, improvements, or bug reports. Note that Wrapt is a mature project, and the primary focus is on maintaining compatibility.

## Testing

For information about running tests, see [TESTING.md](TESTING.md).

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation**: [https://wrapt.readthedocs.io/](https://wrapt.readthedocs.io/)
*   **PyPI**: [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues**: [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog**: [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)
*   **GitHub Repository**: [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)

## Related Blog Posts

Explore a series of blog posts for deeper insights into Wrapt's design and implementation:
(List of blog posts from original README)
```