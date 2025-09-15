# Wrapt: Robust Python Decorators and Wrappers for Enhanced Code

**Wrapt** is the go-to Python module for building powerful and reliable decorators, function wrappers, and monkey patching utilities. ([View on GitHub](https://github.com/GrahamDumpleton/wrapt))

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of Wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (including classmethods and staticmethods), and classes.
*   **Transparent Object Proxies:** Provides advanced wrapping capabilities for complex scenarios.
*   **Safe Monkey Patching:** Offers utilities for making runtime modifications without causing unintended side effects.
*   **Optimized Performance:** Includes a C extension for speed, with a Python fallback for environments without a compiler.
*   **Preserves Introspection:** Ensures that decorators maintain function signatures, annotations, and other crucial metadata for better code maintainability.
*   **Thread-Safe Implementations:** Designed for reliable use in multi-threaded applications.

## Installation

Install Wrapt easily using pip:

```bash
pip install wrapt
```

## Quick Start Examples

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

## Documentation and Resources

*   **Comprehensive Documentation:** [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **PyPI:** [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **Issues:** [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome via the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/).

Please note that the project is mature and the primary focus is to ensure compatibility with newer Python versions.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.