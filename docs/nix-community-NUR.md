# Nix User Repository (NUR): Community-Driven Package Sharing

**Share and discover user-contributed Nix packages quickly and easily with NUR!** [Visit the original repository](https://github.com/nix-community/NUR)

Nix User Repository (NUR) is a powerful community-driven meta-repository designed for Nix package management. It allows users to share their custom Nix packages and configurations, extending the capabilities of Nix and NixOS. Unlike Nixpkgs, NUR offers a faster, more decentralized approach to sharing packages, enabling you to access a wider range of software and experimental builds.

## Key Features

*   **Community-Driven:**  Access packages contributed by a large community of Nix users.
*   **Faster Package Availability:** Get access to new packages and configurations sooner.
*   **Decentralized:** Easily share and discover packages without the need for official Nixpkgs review.
*   **Flexible Installation:** Supports installation via flakes, packageOverrides, and direct integration with NixOS and Home Manager.
*   **Automated Updates:** NUR automatically checks repositories and performs evaluation checks to ensure package integrity.
*   **Extensible:** Provides support for NixOS modules, overlays, and library functions.

## Installation

NUR can be integrated into your Nix environment through various methods, including flakes and packageOverrides.  Here's how to get started:

### Using Flakes

Add NUR to your `flake.nix`:

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

You can then use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add NUR to your `packageOverrides`. First, for your login user, add to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add this to your `/etc/nixos/configuration.nix`.  Ensure you also include this in `~/.config/nixpkgs/config.nix` if you plan to use NUR in `nix-env`, `home-manager`, or `nix-shell`.

### Pinning

To ensure stable builds, pin the version of NUR. This is especially important when using `builtins.fetchTarball`.

```nix
builtins.fetchTarball {
  # Get the revision by choosing a version from https://github.com/nix-community/NUR/commits/main
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  # Get the hash by running `nix-prefetch-url --unpack <url>` on the above url
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

After installation, you can use or install packages from the NUR namespace:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

Or:

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or, for NixOS:

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:** NUR repositories are community-contributed, and it is recommended to review expressions before installing them.

### Example: Using a Single Package in a Devshell (Flake)

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
          overlays = [ nur.overlays.default ];
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

### Integrating with NixOS (Flake Example)

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

Integrate with [Home Manager](https://github.com/rycee/home-manager) by adding NUR modules to your `imports`.

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

Discover available packages through:

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined) (search using GitHub)

## Adding Your Own Repository

Contribute your own packages by creating a repository with a `default.nix` file in its top-level directory.  Use the [repository template](https://github.com/nix-community/nur-packages-template) for a prepared directory structure.

**Important:**  Packages should import dependencies from the `pkgs` argument provided and avoid using `with import <nixpkgs> {}`.

Example `default.nix` structure:

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

where `hello-nur` is a directory containing a `default.nix`:

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

Build your packages using `nix-shell` or `nix-build`:

```console
$ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
nix-shell> hello
Hello, NUR!
```

```console
$ nix-build --arg pkgs 'import <nixpkgs> {}' -A hello-nur
```

For development, set a default `pkgs` argument:

```nix
{ pkgs ? import <nixpkgs> {} }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
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
        "<fill-your-repo-name>": {
            "url": "https://github.com/<your-user>/<your-repo>"
        }
    }
}
```

Run `./bin/nur format-manifest` to sort `repos.json`, add it to git, commit and push, and open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a different nix file as root expression

Use the `file` option to specify an alternate root file:

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

### Updating the Lock File

After pushing changes to your repository, update NUR's lock file using the service at [https://nur-update.nix-community.org/](https://nur-update.nix-community.org/):

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Package Updates

Repositories are evaluated on every build triggered by the URL hook. Repository revisions are updated only if the evaluation is successful.  Common evaluation errors include:

*   Incorrect license attributes in metadata.
*   Using built-in fetchers. Use `pkgs.fetch*` instead.

Check the [latest build job](https://github.com/nix-community/NUR/actions) to see if the evaluation was successful.

#### Local Evaluation Check

Test your packages in your `nur-packages/` folder:

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

Enable submodule support in your repository by setting `submodules`:

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

NUR supports more than just packages:

#### NixOS Modules

Place NixOS modules in the `modules` attribute:

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

Modules should be defined as paths.  If a module is NixOS or Home Manager specific, set the `_class` attribute to `"nixos"` or `"home-manager"`.

#### Overlays

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

#### Library Functions

Put reusable nix functions in the `lib` attribute:

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

You can temporarily override repositories with the `repoOverrides` argument:

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

The repo must be a valid package repo.

### Overriding Repositories with Flakes

(Experimental)  Override repositories in two ways:

-   With `packageOverrides`
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
-   With overlay
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

*   Ensure packages build and set `meta.broken = true` if not.
*   Use [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes) meta attributes.
*   Keep repositories slim.
*   Reuse Nixpkgs packages when possible.

Examples of suitable NUR packages:

*   Packages for a small audience
*   Pre-releases
*   Old package versions
*   Automatic package generation (PyPi, CPAN)
*   Software with opinionated patches
*   Experiments

## Contact

Join the conversation and get help on:
*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)