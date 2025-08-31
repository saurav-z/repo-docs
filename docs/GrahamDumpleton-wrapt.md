# Wrapt: The Python Decorator and Wrapping Library

**Wrapt** is your go-to Python module for building robust and reliable decorators, wrappers, and monkey patching solutions, providing enhanced introspection and thread-safety.

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features

*   **Universal Decorators:** Apply decorators to functions, methods, classmethods, staticmethods, and classes seamlessly.
*   **Transparent Object Proxies:** Leverage advanced wrapping techniques for complex scenarios.
*   **Safe Monkey Patching:** Modify runtime behavior with confidence using dedicated utilities.
*   **Optimized Performance:** Benefit from a C extension for speed, with a Python fallback for portability.
*   **Comprehensive Introspection:** Preserve signatures, annotations, and other critical metadata.
*   **Thread-Safe Implementation:** Ensure stability in concurrent environments.

## Why Choose Wrapt?

Unlike basic decorator implementations, Wrapt is designed for *correctness*. It goes beyond `functools.wraps()` to ensure your decorators behave predictably across diverse use cases, preserving essential Python features.

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

## Documentation

For in-depth information, advanced examples, and detailed usage instructions, explore the official documentation:

*   **[wrapt.readthedocs.io](https://wrapt.readthedocs.io/)**

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome!  Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss potential changes, report bugs, or ask questions.

## Testing

See [TESTING.md](TESTING.md) for details on running tests.

## License

Wrapt is licensed under the BSD License. See the [LICENSE](LICENSE) file for more details.

## Resources

*   **Documentation:** [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **PyPI:** [https://pypi.python.org/pypi/wrapt](https://pypi.python.org/pypi/wrapt)
*   **GitHub Issues:** [https://github.com/GrahamDumpleton/wrapt/issues/](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** [https://wrapt.readthedocs.io/en/latest/changes.html](https://wrapt.readthedocs.io/en/latest/changes.html)
*   **Original Repository:** [https://github.com/GrahamDumpleton/wrapt](https://github.com/GrahamDumpleton/wrapt)

## Related Blog Posts

Explore a series of blog posts that delve into the design and implementation of Wrapt:
(Links to blog posts from original README.)