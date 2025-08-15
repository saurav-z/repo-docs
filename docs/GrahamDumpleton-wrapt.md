# wrapt: Python Decorators, Wrappers, and Monkey Patching - Enhanced & Optimized

**wrapt** is a powerful Python module that simplifies the creation of decorators, wrappers, and monkey patching solutions, ensuring correctness and optimal performance. ([View on GitHub](https://github.com/GrahamDumpleton/wrapt))

[![PyPI](https://img.shields.io/pypi/v/wrapt.svg?logo=python&cacheSeconds=3600)](https://pypi.python.org/pypi/wrapt)
[![Documentation](https://img.shields.io/badge/docs-wrapt.readthedocs.io-blue.svg)](https://wrapt.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

## Key Features of wrapt

*   **Universal Decorators:** Works seamlessly with functions, methods (including classmethods and staticmethods), and classes.
*   **Transparent Object Proxies:** Enables advanced wrapping techniques for complex scenarios.
*   **Safe Monkey Patching:** Provides utilities for safe and reliable runtime modifications of code.
*   **Optimized Performance:** Includes a C extension module for speed-critical components, with a Python fallback.
*   **Introspection Preservation:** Preserves essential information like signatures and annotations for robust decorators.
*   **Thread-Safe Implementations:** Ensures your decorators are safe to use in multi-threaded environments.

## Installation

Install wrapt using pip:

```bash
pip install wrapt
```

## Examples

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

## Documentation & Resources

*   **Comprehensive Documentation:** [wrapt.readthedocs.io](https://wrapt.readthedocs.io/)
*   **PyPI:** [PyPI](https://pypi.python.org/pypi/wrapt)
*   **GitHub Issues:** [GitHub Issues](https://github.com/GrahamDumpleton/wrapt/issues/)
*   **Changelog:** [Changelog](https://wrapt.readthedocs.io/en/latest/changes.html)

## Supported Python Versions

*   Python 3.8+
*   CPython
*   PyPy

## Contributing

Contributions are welcome! Please use the [GitHub issue tracker](https://github.com/GrahamDumpleton/wrapt/issues/) to discuss any changes or report bugs.  Note that wrapt is now a mature project, and the focus is on maintaining compatibility.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Related Blog Posts
This repository also contains a series of blog posts explaining the design and implementation of wrapt.  See the original README for links to those posts.