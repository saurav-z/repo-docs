<picture>
  <source media="(prefers-color-scheme: dark)" srcset="header_dark.jpg">
  <img src="header.jpg" alt="Kindle Comic Converter Header" width="400">
</picture>

# Kindle Comic Converter (KCC)

**Optimize your comics and manga for e-readers with Kindle Comic Converter, ensuring a perfect reading experience!** ([Original Repo](https://github.com/ciromattia/kcc))

[![GitHub release](https://img.shields.io/github/release/ciromattia/kcc.svg)](https://github.com/ciromattia/kcc/releases)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ciromattia/kcc/docker-publish.yml?label=docker%20build)](https://github.com/ciromattia/kcc/pkgs/container/kcc)
[![Github All Releases](https://img.shields.io/github/downloads/ciromattia/kcc/total.svg)](https://github.com/ciromattia/kcc/releases)

Kindle Comic Converter (KCC) is a powerful, cross-platform tool designed to optimize your comics and manga for e-ink e-readers, delivering a superior reading experience. It ensures your favorite comics display in full-screen without distracting margins, with proper fixed layout support.

## Key Features

*   **Broad Format Support:** Converts JPG, PNG, GIF, and PDF files, as well as files within archives (CBZ, CBR, CB7, 7Z, ZIP, RAR).
*   **Versatile Output:** Generates MOBI/AZW3, EPUB, KEPUB, CBZ, and PDF files. PDF output is now directly supported for reMarkable devices, optimizing compatibility.
*   **E-ink Optimization:** Includes image processing steps tailored for e-ink screens, enhancing contrast and readability, resulting in reduced file sizes without sacrificing quality.
*   **Device-Specific Profiles:** Offers profiles for a wide range of e-readers, including Kindle, Kobo, and reMarkable, guaranteeing optimal display.
*   **Resolution Scaling & Cropping:** Automatically downscales images to your device's screen resolution and removes unnecessary margins.
*   **Manga Support:** Correctly handles right-to-left reading and page splitting for manga.
*   **User-Friendly Interface:** Features an intuitive GUI built with Qt6, with tooltips for each option to make conversion easy.

## Why Use KCC?

KCC addresses common formatting issues found on e-readers, such as:

*   Faded black levels
*   Unnecessary margins
*   Incorrect page orientation
*   Misaligned two-page spreads

## NEW: PDF Support for reMarkable

KCC now supports PDF output for direct conversion to reMarkable devices! When using a reMarkable profile (Rmk1, Rmk2, RmkPP), the format automatically defaults to PDF for optimal compatibility with your device's native PDF reader.

## Downloads

Get the latest releases from: [https://github.com/ciromattia/kcc/releases](https://github.com/ciromattia/kcc/releases)

Choose the appropriate download for your operating system:

*   `KCC_\*.\*.\*.exe` (Windows)
*   `kcc_macos_arm_\*.\*.\*.dmg` (Mac with Apple Silicon)
*   `kcc_macos_i386_\*.\*.\*.dmg` (Mac with Intel chip macOS 12+)

## FAQ

Find answers to common questions, including:

*   **Output Format Recommendations:**  Choose MOBI for Kindles, CBZ for Kindle DX and Koreader, and KEPUB for Kobo.
*   **Calibre Compatibility:** Direct USB transfer is recommended instead of using Calibre.
*   **File Transfer on macOS:** On macOS, you may need the Amazon USB File Manager for Mac for some Kindle models.
*   **Gamma Correction:** Disable gamma correction (set gamma to 1.0) if images appear too dark.
*   **Right-to-Left Mode:** RTL mode only affects splitting order for CBZ output.
*   **Colors Inverted:** Disable Kindle dark mode
*   **Kindle Scribe macOS Issue:** Use official MTP [Amazon USB File Transfer app](https://www.amazon.com/gp/help/customer/display.html/ref=hp_Connect_USB_MTP?nodeId=TCUBEdEkbIhK07ysFu)

For more detailed information, consult the full FAQ and extensive documentation on the [KCC Wiki](https://github.com/ciromattia/kcc/wiki/).

## Contributing & Support

*   **General Questions & Feedback:**  [Mobileread Forum](http://www.mobileread.com/forums/showthread.php?t=207461)
*   **Technical Issues:** [GitHub Issues](https://github.com/ciromattia/kcc/issues/new)
*   **Contribute:** Fork the repository and submit pull requests.
*   **Donate:** Support the development of KCC:
    *   Ciro Mattia Gonano (founder): [![Donate PayPal](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=D8WNYNPBGDAS2)
    *   Paweł Jastrzębski:  [![Donate PayPal](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=YTTJ4LK2JDHPS)  [![Donate Bitcoin](https://img.shields.io/badge/Donate-Bitcoin-green.svg)](https://jastrzeb.ski/donate/)
    *   Alex Xu: [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q41BW8HS)

## Credits

*   Ciro Mattia Gonano, Paweł Jastrzębski, Darodi, Alex Xu
*   DC5e's KindleComicParser (inspiration)
*   And various other contributors as noted in the original README.

## License

KCC is released under the ISC LICENSE; see [LICENSE.txt](./LICENSE.txt) for details.