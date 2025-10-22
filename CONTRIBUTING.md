# Contributing to Floship LLM

Thank you for your interest in contributing to the Floship LLM library!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Floship/floship-llm.git
cd floship-llm
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

We use `black` for code formatting:

```bash
black floship_llm/
```

## Type Checking

We use `mypy` for type checking:

```bash
mypy floship_llm/
```

## Making Changes

1. Create a new branch for your feature or bug fix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure tests pass

3. Commit your changes with a descriptive message:
```bash
git commit -m "Add feature: description"
```

4. Push to GitHub and create a pull request

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## Questions?

Open an issue or reach out to the team at dev@floship.com
