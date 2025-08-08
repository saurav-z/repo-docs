# NUR: The Community-Driven Nix Package Repository

**NUR empowers Nix users with a decentralized and fast way to share and install community-contributed Nix packages.** [Check out the NUR GitHub repository for more details.](https://github.com/nix-community/NUR)

## Key Features

*   **Community-Driven:** Access a vast collection of packages contributed by the Nix community.
*   **Faster Package Sharing:** Share new packages quickly, bypassing the review process of Nixpkgs.
*   **Decentralized:** Supports a more open and distributed approach to package management.
*   **Automated Updates:** Automatically checks repositories and performs evaluation checks to ensure package integrity.
*   **Flexible Installation:** Integrates seamlessly with flakes, package overrides, and NixOS configurations.
*   **Supports NixOS Modules, Overlays & Library Functions:** Add modules, overlays & library functions via the repository's `modules`, `overlays` and `lib` attributes.

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>`.

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

For NixOS, add to `/etc/nixos/configuration.nix`. *Note*: If using NUR in `nix-env`, `home-manager`, or `nix-shell`, you *also* need it in `~/.config/nixpkgs/config.nix`.

### Pinning

Pin the version for faster builds (using `sha256` and `fetchTarball`):

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Use packages from the NUR namespace in `nix-shell`, `nix-env`, or your NixOS configuration.

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

Or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important:*** *NUR does not regularly check for malicious content. Always review expressions before installing.*

### Example: Using a Single Package in a Devshell

Here's a simple example of how to add a single package from NUR to a devshell using `flake.nix`:

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

### Example: Using the Flake in NixOS

Integrate NUR's modules and overlays:

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
        # Example adding package with the NUR nixpkgs overlay
        # ({ pkgs, ... }: {
        #   environment.systemPackages = [ pkgs.nur.repos.mic92.hello-nur ];
        # })
      ];
    };
  };
}
```

### Integrating with Home Manager

Integrate with Home Manager by adding your modules to the `imports` attribute.

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
*   [nur-combined](https://github.com/nix-community/nur-combined/search) on GitHub

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file.  Consider using the [repository template](https://github.com/nix-community/nur-packages-template).
2.  *Do not* import packages with `with import <nixpkgs> {};`. Instead, use the `pkgs` argument provided.
3.  Each repository should return a set of Nix derivations:

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

Example `hello-nur/default.nix`:

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

You can build packages via `nix-shell` or `nix-build`:

```console
$ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
nix-shell> hello
nix-shell> find $buildInputs
```

```console
$ nix-build --arg pkgs 'import <nixpkgs> {}' -A hello-nur
```

For development, set a default `pkgs` value:

```nix
{ pkgs ? import <nixpkgs> {} }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

```console
$ nix-build -A hello-nur
```

Add your repository to NUR's `repos.json`:

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
        "<your-repo-name>": {
            "url": "https://github.com/<your-user>/<your-repo>"
        }
    }
}
```

Run:

```console
$ ./bin/nur format-manifest # ensure repos.json is sorted alphabetically
$ git add repos.json
$ git commit -m "add <your-repo-name> repository"
$ git push
```

Then, open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a Different Root Expression

To use a different file instead of `default.nix`, set the `file` option:

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

### Update NUR's Lock File

After updating your repository, you can trigger an update using:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

Details at [https://github.com/nix-community/nur-update#nur-update-endpoint](https://github.com/nix-community/nur-update#nur-update-endpoint)

### Troubleshooting Package Updates

Ensure your evaluation doesn't have errors. Common issues:

*   Incorrect license attributes.
*   Using `builtins.fetchGit` - use `pkgs.fetchgit` instead.

Check the [latest build job](https://github.com/nix-community/NUR/actions) to see if your evaluation succeeded.

#### Local Evaluation Check

In your `nur-packages/` folder, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task:

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

### Git Submodules

To fetch git submodules, set `submodules`:

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

To make NixOS modules, overlays, and library functions more discoverable, they should be placed in their own namespace within the repository.

#### Providing NixOS modules

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

An example can be found [here](https://github.com/Mic92/nur-packages/tree/master/modules). Modules should be defined as paths, not functions.
A module with no [_class](https://nixos.org/manual/nixpkgs/stable/index.html#module-system-lib-evalModules-param-class) will be assumed to be both a NixOS and Home Manager module. If a module is NixOS or Home Manager specific, the `_class` attribute should be set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd).

#### Providing Overlays

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

Override repositories with the `repoOverrides` argument.

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

### Overriding Repositories with Flake (Experimental)

*With packageOverrides*
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
*With overlay*
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

*   Ensure packages build and set `meta.broken = true` if broken.
*   Provide `meta` attributes as per the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean.
*   Reuse packages from Nixpkgs when applicable.

Examples for packages that could be in NUR:

*   Packages for small audiences
*   Pre-releases
*   Old versions of packages
*   Automatic package sets (e.g., from PyPI or CPAN)
*   Software with opinionated patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)