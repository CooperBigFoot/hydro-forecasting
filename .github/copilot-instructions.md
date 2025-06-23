## General Code Generation

- ALWAYS generate Python code compatible with Python 3.10 or newer.
- ALWAYS generate code that adheres to PEP 8 style guidelines.
- ALWAYS generate readable, maintainable, and consistent Python code.
- ALWAYS ensure generated Python code lines DO NOT exceed 88 characters.
- ALWAYS use 4 spaces for indentation in Python code.

## Imports

- ALWAYS organize import statements according to PEP 8:
    1. Standard library imports.
    2. Related third-party imports.
    3. Local application/library-specific imports.
- Separate each import group with a blank line.
- NEVER use wildcard imports (e.g., `from module import *`).
- ALWAYS use absolute imports (e.g., `from my_package.my_module import my_function`). Relative imports are ONLY permissible if they are simpler and confined within the same package.

## Documentation & Docstrings

- ALWAYS generate docstrings for all public modules, functions, classes, and methods.
- When generating docstrings, if a style (e.g., Google, NumPy, reStructuredText) is evident in the existing codebase, ADAPT to and MAINTAIN that style. If no style is evident, PREFER Google style docstrings.
- Docstrings MUST clearly explain:
  - The purpose of the code.
  - Arguments/parameters, including their types and names.
  - Return value(s), including the type.
  - Any exceptions explicitly raised.
- ALWAYS add inline comments to explain the "why" behind non-obvious code sections. Do NOT comment on the "what" if the code is self-explanatory.

## Type Hinting

- ALWAYS use type hints (PEP 484+) for all function signatures (arguments and return types).
- ALWAYS use type hints for variables where the type is not immediately obvious.
- ALWAYS use built-in generic types (e.g., `list[int]`, `dict[str, float]`) for standard collections (Python 3.9+).
- ALWAYS use the `|` operator for union types (e.g., `int | float`, `str | None`) (Python 3.10+).
- USE the `typing` module for specialized types such as `Callable`, `Any`, `TypeVar`, `NewType`, `Protocol`.
- For optional types, PREFER `X | None` over `typing.Optional[X]`.

## Error Handling

- ALWAYS handle exceptions gracefully.
- ALWAYS be specific in `except` clauses (e.g., `except FileNotFoundError:`).
- NEVER use bare `except:` or `except Exception:` unless re-raising or for very generic top-level handlers that include specific logging of the caught exception.
- ALWAYS use the `logging` module for application-level information, warnings, and errors.
- NEVER use `print()` for logging purposes in library or application code. `print()` is acceptable only in short, throwaway scripts or for direct user output in CLI tools.

## Modularity & Structure

- ALWAYS generate code that follows the DRY (Don't Repeat Yourself) principle. If code is repeated, suggest refactoring it into a function or class.
- Generated functions MUST be small and perform one specific task well.
- Generated classes MUST follow the Single Responsibility Principle.

## Testing (when generating test code)

- When generating tests, ALWAYS generate them for the `pytest` framework.
- Generated test files MUST be named `test_*.py` or `*_test.py`.
- Generated test functions MUST be named `test_*()`.
- If asked to create a new test file, it should be placed in a `tests/` directory relative to the code being tested.

## Code Patterns to AVOID

- NEVER use mutable default arguments (e.g., `def func(a, my_list=[])`). Instead, use `my_list: list | None = None` and initialize within the function body: `if my_list is None: my_list = []`.
- PRIORITIZE readability. If a list/dict comprehension or generator expression becomes complex and hard to understand, generate a simple loop instead.
- NEVER silently suppress errors (e.g., `try: ... except: pass`). Errors MUST be logged or handled appropriately by re-raising.
- When generating or modifying modules, structure the code to PREVENT circular dependencies.