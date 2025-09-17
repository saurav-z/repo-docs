# Arrow: Effortless Dates and Times in Python

**Simplify your Python date and time manipulations with Arrow, a user-friendly library for intuitive date and time handling.**  You can find the original repository [here](https://github.com/arrow-py/arrow).

## Key Features:

*   **Drop-in Replacement:** Fully implements and enhances the datetime type.
*   **Python 3.8+ Compatible:** Supports the latest Python versions.
*   **Timezone Awareness:** Handles timezones and defaults to UTC.
*   **Simplified Creation:** Easy-to-use methods for common input scenarios.
*   **Flexible Shifting:** Shift dates and times with relative offsets, including weeks.
*   **Automatic Formatting & Parsing:**  Intelligent string formatting and parsing.
*   **ISO 8601 Support:** Extensive support for the ISO 8601 standard.
*   **Timezone Conversion:** Seamlessly convert between timezones.
*   **Dateutil, Pytz, & ZoneInfo Compatibility:** Works with popular timezone libraries.
*   **Time Span & Range Generation:**  Create time spans, ranges, floors, and ceilings.
*   **Humanization:**  Human-friendly date and time formatting (e.g., "an hour ago").
*   **Extensible:**  Create your own Arrow-derived types.
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Arrow addresses the usability issues often found when working with Python's built-in `datetime` module, including:

*   Reduced reliance on multiple modules (datetime, time, calendar, etc.).
*   Simplified handling of different date and time types.
*   Streamlined timezone conversions and timestamp operations.
*   Improved timezone default behavior.
*   Enhanced functionality with features like ISO 8601 parsing and humanization.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from an ISO 8601 string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time back one hour
utc = utc.shift(hours=-1)
# Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to US/Pacific time
local = utc.to('US/Pacific')
# Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# Output: 1368303838.970460

# Format the time
local.format()
# Output: '2013-05-11 13:23:58 -07:00'

# Format with a specific pattern
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# Output: '2013-05-11 13:23:58 -07:00'

# Humanize the time
local.humanize()
# Output: 'an hour ago'

# Humanize with a specific locale
local.humanize(locale='ko-kr')
# Output: '한시간 전'
```

## Documentation

For comprehensive documentation, visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Refer to the original documentation to find issues and guidelines for contribution.

## Support Arrow

Support the project through the [Open Collective](https://opencollective.com/arrow).