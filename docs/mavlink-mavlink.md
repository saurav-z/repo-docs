# MAVLink: The Lightweight Communication Protocol for Drones and Ground Control Stations

MAVLink is a powerful, header-only message library designed for seamless communication between unmanned aerial vehicles (UAVs) and ground control stations.  **Dive into the world of drone communication with MAVLink, a field-proven protocol optimized for resource-constrained environments!**

[View the original repository on GitHub](https://github.com/mavlink/mavlink)

## Key Features of MAVLink

*   **Lightweight & Efficient:** Optimized for bandwidth-limited applications, making it ideal for UAV communication.  The C implementation is particularly efficient for systems with limited RAM and flash memory.
*   **Header-Only Library:** Simplifies integration as it doesn't require linking.
*   **Cross-Platform Compatibility:** Supports multiple programming languages and operating systems.
*   **Message-Set Specifications:** Defines communication dialects through XML files, ensuring interoperability.
*   **Extensive Documentation & Support:** Comprehensive documentation and community support are available.

## Getting Started

### Prerequisites

*   Ubuntu LTS 20.04 or 22.04 (instructions provided for other platforms)
*   Python 3 and pip installed
*   Git

### Installation

1.  **Install Dependencies:**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Dependencies:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

### Generating C Headers

Generate C headers for `message_definitions/v1.0/common.xml` using the following command:

```bash
python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
```

### Using with CMake

1.  **Install Headers Locally:** Install the headers in a directory (e.g., `install`):

    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Find Package in CMakeLists.txt:**

    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Configure CMake:**

    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

    For a full example, check the [examples/c](examples/c) directory.

    *Note: MAVLink is a header-only library, and `target_link_libraries` doesn't actually link to a library.*

### Further Resources

*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html) (other platforms)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html) (other languages)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)

## Key Links

*   [Documentation/Website](https://mavlink.io/en/)
*   [Discussion/Support](https://mavlink.io/en/#support)
*   [Contributing](https://mavlink.io/en/contributing/contributing.html)
*   [License](https://mavlink.io/en/#license)