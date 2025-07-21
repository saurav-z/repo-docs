# NUR: Your Community-Driven Repository for Nix Packages

**Need cutting-edge or niche Nix packages? NUR provides a decentralized platform for discovering and installing community-contributed packages.**

[View the original repository](https://github.com/nix-community/NUR)

## Key Features:

*   **Community-Driven:** Access a vast collection of user-contributed Nix packages.
*   **Decentralized:** Easily discover and install packages not found in Nixpkgs.
*   **Rapid Updates:** Get access to new packages and updates faster than traditional channels.
*   **Flexible Installation:** Integrate NUR using flakes, `packageOverrides`, NixOS modules, and Home Manager.
*   **Automated Checks:**  NUR performs evaluation checks before updates, minimizing issues.
*   **Package Search:** Find packages through our [search interface](https://nur.nix-community.org/) and our [nur-combined](https://github.com/nix-community/nur-combined) repository.

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

Then use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

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

For NixOS, add it to `/etc/nixos/configuration.nix`.

### Pinning for Stability

Pin the NUR version for build reproducibility:

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
nix-shell> hello
Hello, NUR!
```

Or:

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or:

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important:*** _Always review package expressions before installing them. NUR is a community-driven repository; responsibility for the content rests with contributors._

## Adding Your Repository

1.  **Create a Repository:** Structure your repository with a `default.nix` file and follow the provided [template](https://github.com/nix-community/nur-packages-template).
2.  **Define Packages:** Each repository should return a set of Nix derivations.
3.  **Add to `repos.json`:**  Update the `repos.json` file in NUR to include your repository's URL.
4.  **Submit a Pull Request:**  Open a pull request to the [NUR repository](https://github.com/nix-community/NUR) with your changes.

### Update Lock File and Evaluation

*   Use the nur-update service for faster updates: `curl -XPOST https://nur-update.nix-community.org/update?repo=<your-repo-name>`
*   Verify your repository builds successfully by checking the [latest build job](https://github.com/nix-community/NUR/actions)

## Advanced Usage

*   **NixOS Modules, Overlays, and Library Functions:** Learn how to define NixOS modules, overlays and reusable library functions within your repository.
*   **Repository Overrides:** Test changes with `repoOverrides` or Flakes before they are published.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if necessary.
*   Supply standard meta attributes.
*   Keep repositories slim.
*   Reuse Nixpkgs packages whenever possible.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)