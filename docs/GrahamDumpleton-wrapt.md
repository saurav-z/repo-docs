# wrapt: Enhance Your Python Code with Robust Decorators and Wrappers

**wrapt** is a powerful Python module designed to simplify and improve the creation of decorators, wrappers, and the process of monkey patching. Find out more at the [original repo](https://github.com/GrahamDumpleton/wrapt).

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features

*   **Universal Decorators:** Works seamlessly with functions, methods, classmethods, staticmethods, and classes.
*   **Transparent Object Proxies:** Provides a robust mechanism for advanced wrapping techniques.
*   **Monkey Patching Utilities:** Safely modify code at runtime.
*   **Optimized Performance:** Includes a C extension for performance-critical operations, with a Python fallback.
*   **Introspection Preservation:** Preserves critical information like signatures and annotations.
*   **Thread-Safe Implementation:** Ensures decorator implementations are safe for concurrent use.

## Why Use wrapt?

**wrapt** goes beyond basic decorator implementations by prioritizing correctness and compatibility. It ensures that your decorators maintain introspectability, signatures, and type checking capabilities, making them reliable in a wide range of Python scenarios.

## Installation

To install wrapt, simply use pip:

```bash
pip install wrapt
```

## Example Usage

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

For comprehensive documentation, examples, and advanced usage patterns, visit:

**[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome! Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to suggest changes, report bugs, or ask questions.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Links

*   **Documentation**: https://wrapt.readthedocs.io/
*   **PyPI**: https://pypi.python.org/pypi/wrapt
*   **Issues**: https://github.com/GrahamDumpleton/wrapt/issues/
*   **Changelog**: https://wrapt.readthedocs.io/en/latest/changes.html