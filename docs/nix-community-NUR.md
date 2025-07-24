# Nix User Repository (NUR): Community-Driven Package Sharing for Nix

**NUR empowers Nix users to share and install community-contributed packages, expanding the Nix ecosystem beyond Nixpkgs.**  Learn more about NUR on its [GitHub repository](https://github.com/nix-community/NUR).

## Key Features:

*   **Community-Driven:** Access a vast library of packages curated by the Nix community.
*   **Decentralized:** Share and discover packages faster than traditional methods.
*   **Automated Checks:** Benefit from automated repository checks and evaluation.
*   **Easy Installation:** Integrate NUR using flakes, `packageOverrides`, and more.
*   **Flexible Usage:** Install packages in various Nix environments (nix-shell, nix-env, NixOS configurations, Home Manager).
*   **Package Discovery:** Find packages through the [Packages search for NUR](https://nur.nix-community.org/) or [nur-combined](https://github.com/nix-community/nur-combined).

## Installation:

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

For NixOS, add it to `/etc/nixos/configuration.nix`:

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

Pin your NUR version for stability using `builtins.fetchTarball` with a specific commit hash:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use:

Install packages from NUR using:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```console
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important Note:**  Always review packages before installing them, as NUR packages are not reviewed by Nixpkgs members.*

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

### Example: Using NUR Packages in NixOS (Flake)

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
      ];
    };
  };
}
```

### Integrating with Home Manager

Integrate NUR modules into your `home-manager` configuration:

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

## Finding Packages:

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository:

1.  Create a repository with a `default.nix` file.  Consider using the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Import dependencies from Nixpkgs via the `pkgs` argument.
3.  Define your packages within a set.
4.  Add your repository details to `repos.json` in the NUR repository.
5.  Run `./bin/nur format-manifest` and commit your changes.
6.  Open a pull request on the NUR repository.

### Customizing Repository Behavior:

*   **`file` option:**  Use a different root expression file with the `file` option in `repos.json`.
*   **Update NUR's lock file:** Use the [nur-update service](https://nur-update.nix-community.org/) after you update your repository.

### Troubleshooting:

*   **Package Updates:** Ensure evaluations pass in the [latest build job](https://github.com/nix-community/NUR/actions) for your repo.
*   **Evaluation Errors:** Check your metadata attributes.  Use `pkgs.fetch*` instead of `builtins.fetch*` functions.

#### Local evaluation check
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

### Advanced Features:

*   **Git Submodules:** Enable submodules with `submodules: true` in `repos.json`.
*   **NixOS Modules, Overlays, and Library Functions:** Structure your repository to support these advanced features.  Place NixOS modules in the `modules` attribute, overlays in `overlays`, and library functions in `lib`.
*   **Repository Overrides:** Test changes before publishing with the `repoOverrides` argument. (Flake Support)

## Contribution Guidelines:

*   Ensure packages build and set `meta.broken = true` if not working.
*   Follow Nixpkgs' meta attribute guidelines.
*   Keep repositories lean.
*   Reuse packages from Nixpkgs where possible.
*   See the README for examples of packages that could be in NUR.

## Contact:

*   Matrix channel: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)