#!/usr/bin/env python3
"""
Neuro-Pulse Project Structure Generator.

Creates all required subdirectories under neuro_pulse/ with
__init__.py and .gitkeep files to ensure proper Python packaging
and git tracking of empty directories.
"""

import os


# All directories to create under the project root
DIRECTORIES = [
    "src",
    "dashboard",
    "models",
    "outputs",
    "data",
    "data/real",
    "data/fake",
    "tests",
]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def create_directory(path: str) -> None:
    """Create a directory if it does not exist.

    Args:
        path: Absolute path to the directory.
    """
    os.makedirs(path, exist_ok=True)
    print(f"  [DIR]  {os.path.relpath(path, PROJECT_ROOT)}/")


def create_file(path: str, content: str = "") -> None:
    """Create a file with optional content if it does not exist.

    Args:
        path: Absolute path to the file.
        content: Content to write (default empty).
    """
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
        print(f"  [FILE] {os.path.relpath(path, PROJECT_ROOT)}")
    else:
        print(f"  [SKIP] {os.path.relpath(path, PROJECT_ROOT)} (already exists)")


def main() -> None:
    """Create the full Neuro-Pulse project structure."""
    print("=" * 50)
    print("  Neuro-Pulse: Creating Project Structure")
    print("=" * 50)
    print()

    # Create each directory with __init__.py and .gitkeep
    for dirname in DIRECTORIES:
        dir_path = os.path.join(PROJECT_ROOT, dirname)
        create_directory(dir_path)

        # Add .gitkeep to track empty directories in git
        gitkeep_path = os.path.join(dir_path, ".gitkeep")
        create_file(gitkeep_path)

        # Add __init__.py for Python package directories
        # (skip data directories which are not Python packages)
        if not dirname.startswith("data") and dirname not in ("outputs", "models"):
            init_path = os.path.join(dir_path, "__init__.py")
            create_file(init_path, f'"""Neuro-Pulse {dirname} package."""\n')

    # Create root-level __init__.py for the neuro_pulse package
    root_init = os.path.join(PROJECT_ROOT, "__init__.py")
    create_file(root_init, '"""Neuro-Pulse: rPPG-based Deepfake Detection System."""\n')

    print()
    print("=" * 50)
    print("  Structure created successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
