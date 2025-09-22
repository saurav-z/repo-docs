# Arrow: Intuitive Date and Time Handling for Python

**Simplify your Python date and time manipulations with Arrow, a human-friendly library that makes working with dates, times, and timestamps a breeze.**  Get started with the original project on [GitHub](https://github.com/arrow-py/arrow).

![Build Status](https://github.com/arrow-py/arrow/workflows/tests/badge.svg?branch=master)
![Coverage](https://codecov.io/gh/arrow-py/arrow/branch/master/graph/badge.svg)
![PyPI Version](https://img.shields.io/pypi/v/arrow.svg)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/arrow.svg)
![License](https://img.shields.io/pypi/l/arrow.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Why Choose Arrow?

Tired of complex datetime modules? Arrow streamlines date and time operations, offering a more intuitive and user-friendly experience than Python's built-in modules by addressing common usability pain points.

## Key Features

*   ✅ **Intuitive API:**  Makes date and time creation, manipulation, and formatting simple.
*   ✅ **Timezone Awareness:**  Handles timezones and UTC by default.
*   ✅ **ISO 8601 Support:**  Robust parsing and formatting for ISO 8601 standards.
*   ✅ **Flexible Shift Method:** Easily offset dates and times, including support for weeks.
*   ✅ **Humanization:**  Get human-readable date and time representations ("an hour ago").
*   ✅ **Time Span Generation:** Create time spans, ranges, floors, and ceilings.
*   ✅ **Locale Support:** Humanize dates and times with a growing list of contributed locales.
*   ✅ **Drop-in Replacement:** Fully implemented and works as a drop-in replacement for datetime.
*   ✅ **Type Hinting:** Full support for PEP 484-style type hints.

## Quick Start

### Installation

Install Arrow using pip:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from an ISO 8601 string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time back by one hour
utc = utc.shift(hours=-1)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to another timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# 1368303838.970460

# Format the date and time
local.format()
# '2013-05-11 13:23:58 -07:00'

# Format with a custom format string
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize the time
local.humanize()
# 'an hour ago'

# Humanize with a locale
local.humanize(locale='ko-kr')
# '한시간 전'
```

## Documentation

For comprehensive information, visit the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  See the project's [GitHub repository](https://github.com/arrow-py/arrow/issues) for details on contributing code, localizations, and more.

## Support Arrow

Consider supporting the project by donating through [Open Collective](https://opencollective.com/arrow).