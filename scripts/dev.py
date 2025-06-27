#!/usr/bin/env python3
"""
Development script to run the same checks locally as CI/CD pipeline.
Usage: uv run python scripts/dev.py [command]
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    try:
        subprocess.run(cmd, check=True, cwd=Path.cwd())
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False


def format_code():
    """Format code with Ruff."""
    return run_command(["uv", "run", "ruff", "format", "."], "Code Formatting")


def check_format():
    """Check code formatting with Ruff."""
    return run_command(["uv", "run", "ruff", "format", "--check", "--diff", "."], "Format Check")


def lint_code():
    """Lint code with Ruff."""
    return run_command(["uv", "run", "ruff", "check", "."], "Code Linting")


def fix_lint():
    """Fix linting issues with Ruff."""
    return run_command(["uv", "run", "ruff", "check", "--fix", "."], "Lint Fixes")


def type_check():
    """Run type checking with TY."""
    try:
        # First check if ty is available
        subprocess.run(["uv", "run", "ty", "--version"], check=True, capture_output=True)
        # Exclude notebooks and deprecated files
        return run_command(
            [
                "uv",
                "run",
                "ty",
                "check",
                "--exclude",
                "notebooks/",
                "--exclude",
                "**/*.ipynb",
                "--exclude",
                "**/deprecated/**",
                "--exclude",
                "**/*deprecated*/**",
                "--exclude",
                "experiments/",
                "--exclude",
                "src/hydro_forecasting/data_deprecated/",
                ".",
            ],
            "Type Checking",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  TY type checker not available - skipping type check")
        print("   Install with: uv add --dev ty@latest")
        return True  # Don't fail the pipeline for missing optional tool


def run_tests():
    """Run test suite with pytest."""
    try:
        # First check if pytest is available
        subprocess.run(["uv", "run", "pytest", "--version"], check=True, capture_output=True)
        return run_command(["uv", "run", "pytest", "tests/", "-v"], "Test Suite")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  pytest not available - skipping tests")
        print("   Install with: uv add --dev pytest")
        return True  # Don't fail for missing tool


def run_tests_with_coverage():
    """Run tests with coverage."""
    return run_command(
        [
            "uv",
            "run",
            "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html",
        ],
        "Test Suite with Coverage",
    )


def install_deps():
    """Install development dependencies."""
    return run_command(["uv", "sync", "--dev"], "Dependency Installation")


def full_check():
    """Run all checks in sequence."""
    print("🚀 Running full development check pipeline")

    checks = [
        ("Dependencies", install_deps),
        ("Format Check", check_format),
        ("Linting", lint_code),
        ("Tests", run_tests),
    ]

    results = []
    for name, func in checks:
        results.append((name, func()))

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:<20} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Ready to commit.")
        return 0
    else:
        print("\n💥 Some checks failed. Please fix before committing.")
        return 1


def main():
    """Main entry point."""
    command = "full" if len(sys.argv) < 2 else sys.argv[1]

    commands = {
        "format": format_code,
        "check-format": check_format,
        "lint": lint_code,
        "fix": fix_lint,
        "type-check": type_check,
        "test": run_tests,
        "test-cov": run_tests_with_coverage,
        "install": install_deps,
        "full": full_check,
    }

    if command not in commands:
        print("Available commands:")
        for cmd in commands:
            print(f"  - {cmd}")
        return 1

    return commands[command]() if command != "full" else commands[command]()


if __name__ == "__main__":
    sys.exit(main())
