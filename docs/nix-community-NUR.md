# NUR: The Nix User Repository

**Expand your Nix package ecosystem with NUR, a community-driven repository for user-contributed Nix packages and configurations.**

NUR ([Nix User Repository](https://github.com/nix-community/NUR)) provides access to a wealth of user-submitted Nix packages, built from source and offering a faster, more decentralized way to share and install software compared to Nixpkgs.

## Key Features

*   **Community-Driven:** Discover and install packages contributed by the Nix community.
*   **Rapid Package Availability:** Get access to new packages and updates quickly.
*   **Flexible Installation:** Integrate NUR using flakes, `packageOverrides`, or NixOS configurations.
*   **Automatic Updates:**  NUR automatically checks and validates repositories for updates.
*   **NixOS Modules & Overlays:** Supports NixOS modules, overlays, and library functions for advanced configuration.
*   **Package Search:** Easily find packages using the [Packages search for NUR](https://nur.nix-community.org/) or via [nur-combined](https://github.com/nix-community/nur-combined)

## Installation

### Using Flakes

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

For NixOS, add the following to `/etc/nixos/configuration.nix`:

```nix
{
  nixpkgs.config.packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning for Stability

Pin the NUR version to ensure consistent builds.

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Security Note:** Always review packages before installing them, as NUR packages are not reviewed by Nixpkgs maintainers.*

### Integrating with Home Manager

Integrate with Home Manager by adding modules to the `imports` attribute.

```nix
let
  nur-no-pkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {};
in
{
  imports = lib.attrValues nur-no-pkgs.repos.moredhel.hmModules.rawModules;

  services.unison = {
    enable = true;
    profiles = {
      org = {
        src = "/home/moredhel/org";
        dest = "/home/moredhel/org.backup";
        extraArgs = "-batch -watch -ui text -repeat 60 -fat";
      };
    };
  };
}
```

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file.  Use the [repository template](https://github.com/nix-community/nur-packages-template) as a starting point.

    *   Avoid importing packages with `with import <nixpkgs> {};`.
    *   Use the `pkgs` argument to import dependencies from Nixpkgs.
    *   Each repository should return a set of Nix derivations.

2.  Add your repository details to `repos.json`.

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages"
        },
        "<fill-your-repo-name>": {
            "url": "https://github.com/<your-user>/<your-repo>"
        }
    }
}
```

3.  Run:

    ```bash
    ./bin/nur format-manifest # ensure repos.json is sorted alphabetically
    git add repos.json
    git commit -m "add <your-repo-name> repository"
    git push
    ```

4.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a Different Root Expression

To use a file other than `default.nix`, use the `file` option in `repos.json`.

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "file": "subdirectory/default.nix"
        }
    }
}
```

### Updating the NUR Lock File

Use [https://nur-update.nix-community.org/](https://nur-update.nix-community.org/) to update the NUR lock file after changes.

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Why are my NUR packages not updating?

Common evaluation errors:

*   Incorrect license attributes.
*   Using built-in fetchers instead of `pkgs.fetch*`.

Check the [latest build job](https://github.com/nix-community/NUR/actions) to identify any errors.

### Git Submodules

Enable submodules by setting `submodules`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "submodules": true
        }
    }
}
```

### NixOS Modules, Overlays, and Library Functions

Organize these components within your repository:

*   **NixOS Modules:** Place in the `modules` attribute.
*   **Overlays:** Place in the `overlays` attribute.
*   **Library Functions:** Place in the `lib` attribute.

## Overriding Repositories

Override repositories using the `repoOverrides` argument in your `packageOverrides` or with a flake overlay.

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
        ## remote locations are also possible:
        # mic92 = import (builtins.fetchTarball "https://github.com/your-user/nur-packages/archive/main.tar.gz") { inherit pkgs; };
      };
    };
  };
}
```

## Contribution Guidelines

*   Ensure packages build and set `meta.broken` appropriately.
*   Provide standard `meta` attributes as defined in the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean for efficient downloading and evaluation.
*   Leverage Nixpkgs packages whenever possible.

## Examples for packages that could be in NUR:

*   Packages for niche audiences
*   Pre-releases
*   Older versions of packages
*   Automatic package sets (e.g., from PyPI or CPAN)
*   Software with custom patches
*   Experimental software

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)