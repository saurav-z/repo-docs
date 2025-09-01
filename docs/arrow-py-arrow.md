# Arrow: Effortless Date and Time Handling in Python

**Simplify your Python date and time operations with Arrow, a user-friendly library that makes working with dates and times intuitive and efficient.** Learn more and contribute at the original repository: [https://github.com/arrow-py/arrow](https://github.com/arrow-py/arrow)

[![Build Status](https://github.com/arrow-py/arrow/workflows/tests/badge.svg?branch=master)](https://github.com/arrow-py/arrow/actions?query=workflow%3Atests+branch%3Amaster)
[![Coverage](https://codecov.io/gh/arrow-py/arrow/branch/master/graph/badge.svg)](https://codecov.io/gh/arrow-py/arrow)
[![PyPI Version](https://img.shields.io/pypi/v/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![License](https://img.shields.io/pypi/l/arrow.svg)](https://pypi.python.org/pypi/arrow)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Arrow is a Python library designed to provide a more human-friendly and intuitive way to work with dates, times, and timestamps.  It simplifies the complexities of the standard `datetime` module and offers a more streamlined and user-friendly experience. Inspired by `moment.js` and `requests`, Arrow makes date and time manipulation and formatting a breeze.

## Key Features

*   **Drop-in Replacement:**  Fully implements and enhances the built-in `datetime` type.
*   **Python 3.8+ Compatibility:**  Supports modern Python versions.
*   **Timezone-Aware and UTC by Default:**  Handles timezones intelligently.
*   **Simplified Creation:**  Easy-to-use options for creating dates and times from various inputs.
*   **Flexible Time Manipulation:** Offers a robust `shift` method for relative date/time adjustments, including weeks.
*   **Automatic Formatting and Parsing:**  Intelligently formats and parses strings.
*   **ISO 8601 Support:**  Full support for the ISO 8601 standard.
*   **Timezone Conversion:**  Seamless conversion between timezones.
*   **Integration with Other Libraries:**  Supports `dateutil`, `pytz`, and `ZoneInfo` tzinfo objects.
*   **Time Span and Range Generation:** Creates time spans, ranges, floors, and ceilings from microseconds to years.
*   **Human-Friendly Formatting:**  Humanizes dates and times with support for multiple locales.
*   **Extensible:** Supports custom Arrow-derived types.
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Traditional Python date and time handling can involve multiple modules and complex timezone conversions. Arrow solves this by:

*   Reducing the number of modules needed for common tasks.
*   Simplifying data type management (e.g. dates, datetimes, etc.)
*   Making timezone conversions and timestamp manipulations easier.
*   Providing more intuitive and readable code.
*   Offering functionality gaps missing in the built-in `datetime` module, such as ISO 8601 parsing, timespans, and humanization.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a datetime string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time by one hour
utc = utc.shift(hours=-1)
# Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# Output: 1368303838.970460

# Format the time
local.format()
# Output: '2013-05-11 13:23:58 -07:00'

# Format the time using a custom pattern
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# Output: '2013-05-11 13:23:58 -07:00'

# Humanize the time
local.humanize()
# Output: 'an hour ago'

# Humanize with a different locale
local.humanize(locale='ko-kr')
# Output: '한시간 전'
```

## Documentation

For comprehensive documentation, visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome! Please review the [contributing guidelines](https://github.com/arrow-py/arrow/blob/master/CONTRIBUTING.md) and start by tackling an issue on the [issue tracker](https://github.com/arrow-py/arrow/issues).  Issues marked with the `"good first issue" label <https://github.com/arrow-py/arrow/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_ may be a great place to start!

## Support Arrow

Consider supporting the project through [Open Collective](https://opencollective.com/arrow) to help maintain and improve Arrow.