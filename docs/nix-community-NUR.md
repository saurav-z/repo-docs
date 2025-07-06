# NUR: The Community-Driven Nix Package Repository

**NUR empowers you to access and share Nix packages faster, more flexibly, and collaboratively.** Find it on [GitHub](https://github.com/nix-community/NUR).

## Key Features

*   **Community-Driven:** Access packages contributed by the Nix community.
*   **Decentralized Package Sharing:** Share your own packages and updates quickly.
*   **Automated Checks:**  NUR automatically validates repositories before updates.
*   **Flexible Installation:** Supports flakes, `packageOverrides`, and direct integration.
*   **NixOS Modules & Overlays:** Easily integrate NUR packages into your NixOS configurations.
*   **Home Manager Support:** Seamlessly add NUR modules to Home Manager configurations.
*   **Comprehensive Search:**  Find packages using the NUR package search and `nur-combined` repository.
*   **Repository Templates:** Start building your packages with ease using the provided templates.

## Installation

### Using Flakes

Include NUR in your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
};
```

Then, either the overlay (`overlays.default`) or `legacyPackages.<system>` can be used.

### Using `packageOverrides`

Add the following to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to your `/etc/nixos/configuration.nix`:

```nix
{
  nixpkgs.config.packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning

Pin the version using `builtins.fetchTarball` with a specific SHA256 for reproducibility.

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages using `nix-shell`, `nix-env`, or in your NixOS configuration.

```console
$ nix-shell -p nur.repos.mic92.hello-nur
```
```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```
```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important:  Review packages before installation, as NUR does not regularly check for malicious content.***

##  Examples & Integration

*   **Devshell Example:**  Use a single package in a devshell.
*   **NixOS Integration:**  Integrate NUR overlays and modules in your NixOS configurations.
*   **Home Manager Integration:** Import NUR modules to your Home Manager setup

## Finding Packages

*   **Packages search for NUR:** [Packages search for NUR](https://nur.nix-community.org/)
*   **nur-combined:** [nur-combined](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file. Consider using the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Each repository should return a set of Nix derivations.
3.  Add your repository information to `repos.json` in the NUR repository.
4.  Run `./bin/nur format-manifest` and commit the changes.
5.  Open a pull request to the NUR repository.
6.  Use the NUR update service [https://nur-update.nix-community.org/](https://nur-update.nix-community.org/) to update the lock file after pushing updates.

##  Advanced Usage

*   **Different root expression:** Using a different nix file as root expression with the file option.
*   **Git submodules:** Enable git submodule support in repositories.
*   **NixOS Modules, Overlays, & Library Functions:** Structure your repository for advanced functionality.
*   **Overriding Repositories:** Test changes before publishing using `repoOverrides`.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if broken.
*   Provide standard `meta` attributes from the Nixpkgs manual.
*   Keep repositories lean.
*   Reuse packages from Nixpkgs.

## Contact

Join us on Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) and Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/).