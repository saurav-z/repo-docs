# MAVLink: The Lightweight Communication Protocol for Drones and Ground Control Stations

MAVLink is a powerful, lightweight, and header-only message library that facilitates seamless communication between drones, ground control stations, and other unmanned systems. Learn more at the [original repository](https://github.com/mavlink/mavlink).

## Key Features of MAVLink

*   **Lightweight and Efficient:** Designed for resource-constrained systems, optimized for limited bandwidth, RAM, and flash memory.
*   **Header-Only Library:** Simplifies integration and minimizes dependencies.
*   **Cross-Platform Support:** Compatible with a wide range of programming languages.
*   **XML-Based Message Definitions:** Uses XML files to define message sets ("dialects") for easy customization and expansion.
*   **Field-Proven:** Deployed in numerous products, ensuring reliability and interoperability.
*   **Extensive Documentation & Community Support:** Provides comprehensive resources for developers, including documentation, discussion forums, and contribution guidelines.

## Getting Started with MAVLink (C Example)

Here's how to get started with MAVLink using C on Ubuntu:

1.  **Install Dependencies:**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Requirements:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Headers:**

    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

5.  **Use with CMake:** Install headers locally. Then, in your `CMakeLists.txt` find the package:

    ```cmake
    find_package(MAVLink REQUIRED)

    add_executable(my_program my_program.c)

    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

    And pass the local install directory to cmake (adapt to your directory structure):
     ```bash
     cd ../my_program
     cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
     ```

    Check the `examples/c` directory for a full example.

    *Note: Even though `target_link_libraries` is used in cmake, it doesn't actually "link" to MAVLink as it's just a header-only library.*

## Additional Resources

*   **Documentation/Website:** [https://mavlink.io/en/](https://mavlink.io/en/)
*   **Discussion/Support:** [https://mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [https://mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [https://mavlink.io/en/#license](https://mavlink.io/en/#license)