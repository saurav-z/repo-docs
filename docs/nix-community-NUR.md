# Nix User Repository (NUR): Expand Your Nix Ecosystem

**NUR empowers the Nix community to share and install packages faster with user-contributed repositories.** Learn more about NUR on its [original repository](https://github.com/nix-community/NUR).

## Key Features

*   **Community-Driven:** Access a wide range of packages contributed and maintained by the Nix community.
*   **Decentralized:**  Share and install packages quickly, bypassing the formal review process of Nixpkgs.
*   **Automatic Evaluation:**  NUR automatically checks and validates repositories before updating, ensuring package integrity.
*   **Flexible Installation:** Integrate NUR into your Nix setup using flakes, `packageOverrides`, or direct package installation.
*   **Easy Package Discovery:** Find packages using the [NUR package search](https://nur.nix-community.org/) or the [nur-combined](https://github.com/nix-community/nur-combined) repository.
*   **Supports NixOS Modules, Overlays, and Library Functions:** Beyond packages, NUR supports other Nix expressions and integrates easily with NixOS.

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add NUR to your `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add to `/etc/nixos/configuration.nix`:

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

For stable installations, pin the NUR version using `fetchTarball` with a `sha256` hash:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages from NUR using `nix-shell`, `nix-env`, or your `configuration.nix`:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
```

or

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:**  NUR is community-driven.  *Always check the expressions before installing packages*.

## Examples

### Single Package in a Devshell (Flake)

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

### Using NUR in NixOS (Flake)

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
      ];
    };
  };
}
```

### Integrating with Home Manager

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

1.  **Create a Repository:** Create a Git repository with a `default.nix` file at the root.  Use the [repository template](https://github.com/nix-community/nur-packages-template) for guidance.
2.  **Nix Expression:**  Your `default.nix` should return a set of Nix derivations, taking `pkgs` as an argument (from Nixpkgs):

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

3.  **Package Definition:** Example of `hello-nur/default.nix`:

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

4.  **Testing:**  Test your packages using `nix-shell` or `nix-build`.
5.  **Add to NUR:**  Add your repository details to `repos.json` in the main NUR repository:

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

6.  **Submit a PR:**  Create a pull request with your changes to `repos.json`.

7.  **Optional: Use Different Nix File:** For a different root expression, set the `file` option:

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

8.  **Optional: Submodules:** Enable submodule support with `submodules = true`.

### Update NUR's lock file after updating your repository

After you have pushed an update to your repository, you can use our service at https://nur-update.nix-community.org/
```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```
Check out the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for further details

## Why are my NUR packages not updating?

*   Ensure your repository evaluates without errors. Check the [latest build job](https://github.com/nix-community/NUR/actions) on the NUR repository.
*   Common errors:  incorrect license attributes, using built-in fetchers instead of `pkgs.fetch*`.

### Local evaluation check

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

## Overriding Repositories

You can test local changes using `repoOverrides`.

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
      };
    };
  };
}
```

**Experimental Flake Overrides:**

Can use `packageOverrides` or `overlays`. The repo must contain a `flake.nix` file and a `default.nix`:

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

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if they do not.
*   Use standard `meta` attributes.
*   Keep repositories small and leverage Nixpkgs where possible.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)