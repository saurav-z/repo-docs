# Arrow: Python's Human-Friendly Date and Time Library

**Simplify your Python date and time manipulations with Arrow, a powerful and intuitive library designed for ease of use.**  [View the project on GitHub](https://github.com/arrow-py/arrow)

Arrow provides a streamlined and human-friendly approach to working with dates, times, and timestamps in Python, addressing the complexities of the standard `datetime` module. It's inspired by libraries like `moment.js` and `requests`, making date and time operations more intuitive.

## Key Features

*   **Intuitive and Easy to Use:**  Create, manipulate, format, and convert dates and times with minimal code.
*   **Drop-in Replacement for Datetime:** Fully implements and extends the functionality of Python's `datetime` objects.
*   **Timezone-Aware by Default:** Handles timezones with ease, defaulting to UTC.
*   **Simple Creation Options:** Easily create Arrow objects from common input formats.
*   **Flexible Date & Time Shifting:**  Use the `shift` method for relative offsets, including support for weeks.
*   **Automatic String Formatting and Parsing:**  Intelligent handling of date and time strings.
*   **ISO 8601 Support:**  Robust support for the ISO 8601 standard.
*   **Timezone Conversion:** Seamlessly convert between timezones.
*   **Comprehensive Time Span Generation:** Create time spans, ranges, floors, and ceilings for various time frames.
*   **Humanization:**  Generate human-readable date and time representations (e.g., "an hour ago").
*   **Extensible:** Designed for custom Arrow-derived types.
*   **Type Hints:** Full support for PEP 484-style type hints.
*   **Python Version Support:** Compatible with Python 3.8+

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from a string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get timestamp
local.timestamp()
# 1368303838.970460

# Format the date and time
local.format()
# '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize the date and time
local.humanize()
# 'an hour ago'
local.humanize(locale='ko-kr')
# '한시간 전'
```

## Documentation

For in-depth information and examples, please visit the official documentation:  [arrow.readthedocs.io](https://arrow.readthedocs.io)

## Contributing

Contributions are warmly welcomed!  Please refer to the [GitHub repository](https://github.com/arrow-py/arrow) for guidelines on contributing code, documentation, and localizations.

## Support Arrow

Support the development of Arrow by making a financial contribution through [Open Collective](https://opencollective.com/arrow).