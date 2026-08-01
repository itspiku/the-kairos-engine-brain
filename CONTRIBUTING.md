# Contributing to The Kairos Engine

Thank you for your interest in contributing to **The Kairos Engine** — an AI cognitive autopilot for BVLOS medical drone operations in the Himalayas. Contributions that improve safety, reliability, and performance are warmly welcomed.

## Table of Contents
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Architecture Overview](#architecture-overview)
- [Safety Guidelines](#safety-guidelines)

---

## Development Setup

### Prerequisites
- Python 3.10 or 3.11
- Git
- (Optional) CUDA-capable GPU for LLM inference
- (Optional) Docker

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/The-Khumbu-Engine---Brain.git
cd The-Khumbu-Engine---Brain

# Install all dependencies
make install

# Build the XGBoost crash predictor model
make train-model

# Run the test suite
make test
```

### Install for Development

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with all development extras (ruff, mypy, pytest).

---

## Code Style

### Formatter & Linter
We use **[ruff](https://docs.astral.sh/ruff/)** for both linting and formatting.

```bash
# Auto-format all code
make format

# Check for lint issues
make lint
```

### Type Hints
All production code in `src/` **must** include type hints. Run mypy to check:

```bash
make typecheck
```

### Docstrings
Every public function, class, and module must have a Google-style docstring:

```python
def my_function(param: str) -> dict:
    """
    Brief one-line description.

    Longer description if needed.

    Args:
        param: Description of the parameter.

    Returns:
        Description of the return value.

    Raises:
        KairosError: If something goes wrong.
    """
```

### Import Convention
All imports must use the `src.` prefix (the project is run from the root directory):

```python
# Correct
from src.core.exceptions import KairosError
from src.core.types import FlightAction

# Incorrect
from core.exceptions import KairosError
```

---

## Testing Requirements

### Running Tests

```bash
# Full test suite
make test

# Fast tests only (no model required)
make test-fast

# With coverage report
make test-cov
```

### Requirements for New Code
- **All new functions** must have corresponding unit tests.
- **Tests must NOT require the Gemma 4 GGUF model** to be loaded — mock the LLM.
- **Integration tests** (in `tests/test_integration.py`) may require the XGBoost model but not the GGUF.
- Tests must use Python's `unittest` framework.
- Target >80% code coverage for `src/` modules.

### Test File Naming Convention
- `tests/test_<module_name>.py` — matches the source module being tested
- Each test class should inherit from `unittest.TestCase`
- Each test method must start with `test_`

### Example Test

```python
class TestMyFeature(unittest.TestCase):

    def test_normal_operation(self):
        from src.my_module import my_function
        result = my_function("input")
        self.assertIsInstance(result, dict)
        self.assertIn("key", result)

    def test_error_handling(self):
        from src.my_module import my_function
        from src.core.exceptions import KairosError
        with self.assertRaises(KairosError):
            my_function(None)
```

---

## Submitting a Pull Request

### Workflow

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines above.

3. **Add tests** for your changes.

4. **Run the full CI check locally**:
   ```bash
   make ci
   ```

5. **Commit** with a clear, descriptive message:
   ```
   feat(routing): add wind-aware edge cost adjustment for crosswinds

   - Modify calculate_edge_cost() to factor in crosswind component
   - Add test_edge_cost_crosswind_penalty to test_routing.py
   - Update dijkstra.py docstring
   ```

6. **Push** and open a Pull Request against `main`.

### PR Checklist
- [ ] Code follows the style guide (ruff passes)
- [ ] Type hints added (mypy passes)
- [ ] Tests added for new functionality
- [ ] All tests pass (`make test`)
- [ ] Docstrings added/updated
- [ ] No GGUF model file committed
- [ ] No sensitive credentials committed

### Commit Message Format
We follow Conventional Commits:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Tests only |
| `refactor:` | Code restructuring |
| `perf:` | Performance improvement |
| `chore:` | Build/config changes |

---

## Architecture Overview

```
src/
├── config.py               # Constants and thresholds
├── parser.py               # Gemma 4 DSL tool call parser
├── brain.py                # GemmaBrain/KairosBrain convenience wrappers
├── core/
│   ├── engine.py           # Master orchestrator (main entry point)
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── telemetry.py        # TelemetryReport dataclass
│   └── types.py            # FlightAction, RiskCategory, CognitiveResponse
├── models/
│   ├── llm.py              # GemmaLLMEngine (llama-cpp-python wrapper)
│   └── risk_classifier.py  # XGBoost singleton risk predictor
├── routing/
│   ├── dijkstra.py         # NetworkX graph-based pathfinder
│   └── spatial.py          # Haversine, elevation, airspace utilities
├── telemetry/
│   └── anomaly_detector.py # In-flight anomaly detection & action recommendation
├── tools/
│   ├── definitions.py      # OpenAI-compatible JSON tool schemas
│   ├── executor.py         # Tool routing with real/mock fallback
│   ├── implementations.py  # Real tool implementations (APIs + physics models)
│   └── mocks.py            # Mock fallbacks for testing/offline use
├── utils/
│   ├── formatter.py        # LLM prompt formatters and report builders
│   └── logger.py           # KairosLogger (structured logging + audit trail)
└── visualization/
    └── map_renderer.py     # Folium interactive GIS map generation
```

---

## Safety Guidelines

> ⚠️ **This project is used in life-critical medical drone operations. Safety is non-negotiable.**

1. **Never reduce thresholds** for `CRITICAL_RISK_THRESHOLD` or `HIGH_RISK_THRESHOLD` without validated test data.
2. **Always add graceful fallbacks** for any external API call (Open-Meteo, Open-Elevation). The drone must function offline.
3. **Never remove the SLA check**. All in-flight decisions must enforce the 2-second `MAX_LATENCY_SLA_SEC` limit.
4. **Audit all decisions**. Every flight decision must be logged via `log_decision()` to `logs/kairos_audit.jsonl`.
5. **Test anomaly detection thoroughly** before modifying `anomaly_detector.py`. Emergency scenarios must always produce conservative (safe) action recommendations.

---

## Questions?

Open an issue or start a discussion on GitHub. For security vulnerabilities, email the maintainers directly.
