"""Weight-free example. The gradients are explicitly synthetic, not Boltz output."""

from molcraft.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", "--output", "outputs/example"]))
