# Arrow: Effortless Date and Time Handling in Python

**Simplify your Python date and time operations with Arrow, a human-friendly library designed to make working with dates and times a breeze.**  [View the original repository on GitHub](https://github.com/arrow-py/arrow).

[![Build Status](https://github.com/arrow-py/arrow/workflows/tests/badge.svg?branch=master)](https://github.com/arrow-py/arrow/actions?query=workflow%3Atests+branch%3Amaster)
[![Coverage](https://codecov.io/gh/arrow-py/arrow/branch/master/graph/badge.svg)](https://codecov.io/gh/arrow-py/arrow)
[![PyPI Version](https://img.shields.io/pypi/v/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![License](https://img.shields.io/pypi/l/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features

*   **Intuitive datetime replacement:**  A fully implemented drop-in replacement for Python's `datetime`.
*   **Broad Python Version Support:** Supports Python 3.8 and newer.
*   **Timezone-aware by Default:**  Works with timezones intelligently, UTC by default.
*   **Simplified Creation:** Easy-to-use methods for creating dates and times from common inputs.
*   **Flexible Time Manipulation:** Shift dates and times using relative offsets (e.g., days, weeks).
*   **Intelligent Formatting and Parsing:** Automatic string formatting and parsing for ease of use.
*   **ISO 8601 Support:**  Comprehensive support for the ISO 8601 standard.
*   **Timezone Conversion:**  Seamless timezone conversions.
*   **Integration with Existing Libraries:** Supports `dateutil`, `pytz`, and `ZoneInfo` `tzinfo` objects.
*   **Time Span Generation:** Generate time spans, ranges, floors, and ceilings.
*   **Human-Friendly Output:** Humanize dates and times with locale support (e.g., "an hour ago", "1 hour ago").
*   **Extensible:** Designed for custom Arrow-derived types.
*   **Type Hinting:** Fully compatible with PEP 484-style type hints.

## Why Use Arrow?

Arrow streamlines date and time operations, addressing the usability challenges of the standard Python `datetime` and related modules.  Arrow simplifies:

*   Reducing the number of modules you need to import
*   Simplifying the range of datetime related types used.
*   Making timezone conversions and timestamp operations more user-friendly.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a string
arw = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(arw)
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get UTC time
utc = arrow.utcnow()
print(utc)
# <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time
utc = utc.shift(hours=-1)
print(utc)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to local time
local = utc.to('US/Pacific')
print(local)
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get timestamp
print(local.timestamp())
# 1368303838.970460

# Format the time
print(local.format())
# '2013-05-11 13:23:58 -07:00'

print(local.format('YYYY-MM-DD HH:mm:ss ZZ'))
# '2013-05-11 13:23:58 -07:00'

# Humanize the time
print(local.humanize())
# 'an hour ago'

print(local.humanize(locale='ko-kr'))
# '한시간 전'
```

## Documentation

For detailed information, please visit the full documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Learn how to contribute in the original [README](https://github.com/arrow-py/arrow).

## Support Arrow

If you appreciate Arrow and would like to support its development, consider making a donation via the [Open Collective](https://opencollective.com/arrow).