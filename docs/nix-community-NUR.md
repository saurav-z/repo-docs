# NUR: The Nix User Repository

**Expand your Nix package ecosystem with the Nix User Repository (NUR), a community-driven meta-repository for Nix packages.**

NUR allows you to access a vast collection of user-contributed Nix packages, built from source and not reviewed by Nixpkgs members, expanding the Nix package options and accelerating access to community-created software.  Check out the original repo [here](https://github.com/nix-community/NUR).

## Key Features

*   **Community-Driven:** Access packages created and maintained by the Nix community.
*   **Decentralized:**  Discover and install packages quickly, outside the standard Nixpkgs review process.
*   **Package Discovery:** Easily find packages through the [Packages search for NUR](https://nur.nix-community.org/) or the [nur-combined](https://github.com/nix-community/nur-combined) repository.
*   **Flexible Installation:** Integrate NUR using flakes, `packageOverrides`, or integrate directly into your NixOS configuration.
*   **Automated Checks:** NUR performs automatic checks and evaluations to ensure repository integrity.

## Installation

### Using Flakes

Integrate NUR into your `flake.nix`:

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

Then, use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add NUR to your `packageOverrides`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

*   For your login user, add the above to `~/.config/nixpkgs/config.nix`.
*   For NixOS, add it to `/etc/nixos/configuration.nix`. Remember to add it to `~/.config/nixpkgs/config.nix` to use NUR in `nix-env`, `home-manager`, or `nix-shell`.

### Pinning

Pin versions to prevent cache invalidation and improve build reproducibility:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Use or install packages from the NUR namespace:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
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

***Important Safety Note:**  NUR doesn't regularly scan repositories for malicious content.  Always inspect packages before installing them.*

### Using a Single Package in a Devshell

Add a package from NUR to a devshell defined in a `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nur }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlay ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.nur.repos.mic92.hello-nur ];
        };
      }
    );
}
```

### Using the Flake in NixOS

Integrate NUR modules and overlays into your NixOS configuration:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nur }: {
    nixosConfigurations.myConfig = nixpkgs.lib.nixosSystem {
      # ...
      modules = [
        # Adds the NUR overlay
        nur.modules.nixos.default
        # NUR modules to import
        nur.legacyPackages."${system}".repos.iopq.modules.xraya
        # This adds the NUR nixpkgs overlay.
        # Example:
        # ({ pkgs, ... }: {
        #   environment.systemPackages = [ pkgs.nur.repos.mic92.hello-nur ];
        # })
      ];
    };
  };
}
```

### Integrating with Home Manager

Integrate with Home Manager by adding NUR modules to the `imports` attribute:

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

Search for NUR packages:

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined) repository via [github](https://github.com/nix-community/nur-combined/search).

## Adding Your Own Repository

1.  **Create a Repository:** Create a Git repository with a `default.nix` file at the top level.  Use the [repository template](https://github.com/nix-community/nur-packages-template) as a starting point.
2.  **Package Definition:**  Your `default.nix` should return a set of Nix derivations, *using* `pkgs` as the argument.  *Do not* import packages directly from `<nixpkgs>`.
3.  **Example `default.nix`:**

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

4.  **Example `hello-nur/default.nix`:**

    ```nix
    { stdenv, fetchurl, lib }:

    stdenv.mkDerivation rec {
      pname = "hello";
      version = "2.10";

      src = fetchurl {
        url = "mirror://gnu/hello/${pname}-${version}.tar.gz";
        sha256 = "0ssi1wpaf7plaswqqjwigppsg5fyh99vdlb9kzl7c9lng89ndq1i";
      };

      postPatch = ''
        sed -i -e 's/Hello, world!/Hello, NUR!/' src/hello.c
      '';

      # fails due to patch
      doCheck = false;

      meta = with lib; {
        description = "A program that produces a familiar, friendly greeting";
        longDescription = ''
          GNU Hello is a program that prints "Hello, world!" when you run it.
          It is fully customizable.
        '';
        homepage = https://www.gnu.org/software/hello/manual/;
        changelog = "https://git.savannah.gnu.org/cgit/hello.git/plain/NEWS?h=v${version}";
        license = licenses.gpl3Plus;
        maintainers = [ maintainers.eelco ];
        platforms = platforms.all;
      };
    }
    ```
5.  **Test Your Packages:** Build your packages using `nix-shell` or `nix-build`.

    ```console
    $ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
    nix-shell> hello
    nix-shell> find $buildInputs
    ```

    ```console
    $ nix-build --arg pkgs 'import <nixpkgs> {}' -A hello-nur
    ```

6.  **Optional `pkgs` Argument:**  Set a default value for the `pkgs` argument for development convenience.

    ```nix
    { pkgs ? import <nixpkgs> {} }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```
7.  **Add Your Repo to `repos.json`:**

    ```console
    $ git clone --depth 1 https://github.com/nix-community/NUR
    $ cd NUR
    ```

    Edit `repos.json`:

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

    *   Each URL must point to a Git repository.
    *   Run `./bin/nur format-manifest` to sort `repos.json` alphabetically.
    *   Git add `repos.json`.  *Do not* add `repos.json.lock`.
    *   Commit and push your changes.
8.  **Open a Pull Request:** Create a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

*Repositories should be buildable on Nixpkgs unstable.*

### Using a Different Root Expression

Use the `file` option to specify a different file to load packages from:

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

### Updating NUR's Lock File

Update the NUR lock file faster using the nur-update service at https://nur-update.nix-community.org/ after updating your repository:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

See the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for details.

### Troubleshooting Package Updates

Ensure your repository is evaluated correctly by checking the [latest build job](https://github.com/nix-community/NUR/actions).

Common evaluation errors:

*   Incorrect license attributes.
*   Use of `builtins.fetchGit` (use `pkgs.fetchgit` instead).

#### Local Evaluation Check

In your `nur-packages/` folder, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task

```sh
nix-env -f . -qa \* --meta \
  --allowed-uris https://static.rust-lang.org \
  --option restrict-eval true \
  --option allow-import-from-derivation true \
  --drv-path --show-trace \
  -I nixpkgs=$(nix-instantiate --find-file nixpkgs) \
  -I ./ \
  --json | jq -r 'values | .[].name'
```

On success, this shows a list of your packages

### Git Submodules

Enable Git submodules in your repository definition:

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

Organize NixOS modules, overlays, and library functions within your repository for better discoverability.

#### Providing NixOS Modules

Place NixOS modules within the `modules` attribute:

```nix
{ pkgs }: {
  modules = import ./modules;
}
```

```nix
# modules/default.nix
{
  example-module = ./example-module.nix;
}
```

A module with no [_class](https://nixos.org/manual/nixpkgs/stable/index.html#module-system-lib-evalModules-param-class) will be assumed to be both a NixOS and Home Manager module. If a module is NixOS or Home Manager specific, the `_class` attribute should be set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd).

#### Providing Overlays

Use the `overlays` attribute for overlays:

```nix
# default.nix
{
  overlays = {
    hello-overlay = import ./hello-overlay;
  };
}
```

```nix
# hello-overlay/default.nix
self: super: {
  hello = super.hello.overrideAttrs (old: {
    separateDebugInfo = true;
  });
}
```

#### Providing Library Functions

Put reusable Nix functions in the `lib` attribute:

```nix
{ pkgs }:
with pkgs.lib;
{
  lib = {
    hexint = x: hexvals.${toLower x};

    hexvals = listToAttrs (imap (i: c: { name = c; value = i - 1; })
      (stringToCharacters "0123456789abcdef"));
  };
}
```

## Overriding Repositories

You can override repositories using `repoOverrides` argument.  This allows to test changes before publishing.

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

The repo must be a valid package repo, i.e. its root contains a `default.nix` file.

### Overriding repositories with Flake

**Experimental** Note that flake support is still experimental and might change in future in a backwards incompatible way.

You can override repositories in two ways:

- With packageOverrides
```nix
{
  inputs.nur.url = "github:nix-community/NUR";
  inputs.paul.url = "path:/some_path/nur-paul"; # example: a local nur.repos.paul for development

  outputs = {self, nixpkgs, nur, paul }: {

  system = "x86_64-linux";

  nurpkgs = import nixpkgs { inherit system; };

  ...
  modules = [
       {
         nixpkgs.config.packageOverrides = pkgs: {
            nur = import nur {
              inherit pkgs nurpkgs;
              repoOverrides = { paul = import paul { inherit pkgs; }; };
            };
          };
        }
  ];
  ...
}
```
- With overlay
```nix
{
  modules = [
    {
      nixpkgs.overlays = [
        (final: prev: {
          nur = import nur {
            nurpkgs = prev;
            pkgs = prev;
            repoOverrides = { paul = import paul { pkgs = prev; }; };
          };
        })
      ];
    }
    ...
  ];
}
```

The repo must contain a `flake.nix` file in addition to a `default.nix`:  [flake.nix example](https://github.com/Mic92/nur-packages/blob/master/flake.nix)

## Contribution Guidelines

*   **Build and Metadata:** Ensure packages build and include all standard `meta` attributes.  Set `meta.broken = true` if a package is broken.
*   **Repository Size:** Keep repositories lean to optimize download and evaluation times.
*   **Reuse Nixpkgs:** Favor packages from Nixpkgs whenever possible to leverage the binary cache.

**Ideal NUR Package Candidates:**

*   Packages for a limited audience.
*   Pre-release versions.
*   Older package versions not in Nixpkgs.
*   Automatically generated package sets.
*   Software with specific patches.
*   Experimental software.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)