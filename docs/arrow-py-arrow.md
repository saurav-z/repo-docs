# Arrow: Python's Human-Friendly Date & Time Library

Tired of struggling with dates and times in Python? [Arrow](https://github.com/arrow-py/arrow) simplifies date and time manipulation, making your code cleaner and more readable.

[![Build Status](https://github.com/arrow-py/arrow/workflows/tests/badge.svg?branch=master)](https://github.com/arrow-py/arrow/actions?query=workflow%3Atests+branch%3Amaster)
[![Coverage](https://codecov.io/gh/arrow-py/arrow/branch/master/graph/badge.svg)](https://codecov.io/gh/arrow-py/arrow)
[![PyPI Version](https://img.shields.io/pypi/v/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![License](https://img.shields.io/pypi/l/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features:

*   **Intuitive API:** Easily create, manipulate, and format dates and times.
*   **Timezone Awareness:** Works with timezones and UTC by default.
*   **ISO 8601 Support:** Seamlessly parse and format ISO 8601 strings.
*   **Relative Offsets:** Use the `shift` method for intuitive time manipulation (e.g., shifting by weeks, days, etc.).
*   **Humanization:**  Convert dates and times to human-readable strings (e.g., "an hour ago").
*   **Time Span Generation:**  Generate time spans, ranges, floors, and ceilings.
*   **Localization:** Supports multiple locales for humanization.
*   **Python 3.8+ Support:**  Compatible with the latest Python versions.
*   **Type Hints:** Fully supports PEP 484-style type hints.

## Why Use Arrow?

Arrow addresses the usability issues present in Python's built-in `datetime` module and related libraries, providing a more streamlined and user-friendly experience. Avoid the complexity of multiple modules and types with Arrow's simplified approach.

## Quick Start

### Installation:

```bash
pip install -U arrow
```

### Example Usage:

```python
import arrow

# Create an Arrow object
now = arrow.now()
print(now) # <Arrow [2024-02-29T10:30:00.123456+00:00]>

# Shift time
past = now.shift(days=-7)
print(past) # <Arrow [2024-02-22T10:30:00.123456+00:00]>

# Convert to another timezone
pacific_time = now.to('US/Pacific')
print(pacific_time) # <Arrow [2024-02-29T02:30:00.123456-08:00]>

# Format the date/time
print(pacific_time.format('YYYY-MM-DD HH:mm:ss ZZ')) # 2024-02-29 02:30:00 -08:00

# Humanize
print(past.humanize()) # a week ago
```

## Documentation

For detailed information, visit the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome! Please see the original repo's README for details on contributing code and localizations.

## Support Arrow

Consider supporting the project through the [Open Collective](https://opencollective.com/arrow) to help ensure its continued development.