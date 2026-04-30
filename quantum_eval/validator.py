"""
Three-level validation pipeline for generated Qiskit code.

Level 1: Syntax validation (AST parsing)
Level 2: Execution validation (sandboxed subprocess)
Level 3: Semantic validation (category-specific checks)
"""

import ast
import subprocess
import tempfile
import re
import json
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum

class ValidationLevel(Enum):
    NONE = 0
    SYNTAX = 1
    EXECUTION = 2
    SEMANTIC = 3


@dataclass
class ValidationResult:
    """Result of validation pipeline."""
    valid: bool
    level_passed: ValidationLevel
    error: Optional[str] = None
    warnings: List[str] = None
    execution_time_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# =============================================================================
# DEPRECATED PATTERN DETECTION
# =============================================================================

DEPRECATED_PATTERNS = [
    (r"from qiskit import execute", "Use Sampler/Estimator primitives instead of execute()"),
    (r"from qiskit\.providers\.aer import", "Import from qiskit_aer instead"),
    (r"\.execute\s*\(", "Use Sampler/Estimator primitives instead of .execute()"),
    (r"from qiskit\.tools", "qiskit.tools is deprecated"),
    (r"qiskit\.IBMQ", "Use QiskitRuntimeService instead of IBMQ"),
    (r"from qiskit\.providers\.ibmq", "Use qiskit_ibm_runtime instead"),
    (r"BasicAer", "Use qiskit_aer.AerSimulator instead of BasicAer"),
    (r"Aer\.get_backend", "Use AerSimulator() directly"),
    (r"from qiskit\.aqua", "qiskit.aqua is deprecated, use qiskit_algorithms"),
    (r"from qiskit\.chemistry", "Use qiskit_nature instead"),
    (r"from qiskit\.optimization", "Use qiskit_optimization instead"),
    (r"assemble\s*\(", "assemble() is deprecated, use primitives"),
]

REQUIRED_PATTERNS = {
    "primitives_api": [
        (r"from qiskit\.primitives import|from qiskit_ibm_runtime import", 
         "Primitives API examples must import from qiskit.primitives"),
    ],
    "noise_modeling": [
        (r"from qiskit_aer|AerSimulator", 
         "Noise modeling must use qiskit_aer"),
    ],
}


def check_deprecated_patterns(code: str) -> List[str]:
    """Check for deprecated API patterns. Returns list of issues found."""
    issues = []
    for pattern, message in DEPRECATED_PATTERNS:
        if re.search(pattern, code):
            issues.append(f"Deprecated: {message}")
    return issues


def check_required_patterns(code: str, category: str) -> List[str]:
    """Check that category-specific required patterns are present."""
    issues = []
    if category in REQUIRED_PATTERNS:
        for pattern, message in REQUIRED_PATTERNS[category]:
            if not re.search(pattern, code):
                issues.append(f"Missing required: {message}")
    return issues


# =============================================================================
# TEST STUB DETECTION
# =============================================================================

TEST_STUB_PATTERNS = [
    (r'def\s+test_\w+\s*\(\s*self\b', "Test method detected (def test_*(self))"),
    (r'def\s+setUp\s*\(\s*self\b', "Test lifecycle method (setUp)"),
    (r'def\s+tearDown\s*\(\s*self\b', "Test lifecycle method (tearDown)"),
    (r'class\s+\w+\s*\(\s*(?:\w+\.)*TestCase\s*\)', "TestCase subclass"),
    (r'class\s+\w+\s*\(\s*(?:\w+\.)*QiskitTestCase\s*\)', "QiskitTestCase subclass"),
]


def detect_test_stubs(code: str) -> tuple:
    """Detect memorized test patterns from training data.

    Returns (is_stub: bool, reason: str).
    """
    for pattern, reason in TEST_STUB_PATTERNS:
        if re.search(pattern, code):
            return True, f"Test stub detected: {reason}"
    return False, ""


# =============================================================================
# DEAD CODE DETECTION
# =============================================================================

def detect_dead_code(code: str) -> tuple:
    """Detect code that only defines functions/classes without executing them.

    Returns (is_dead: bool, reason: str).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, ""

    has_definition = False
    has_executable = False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            has_definition = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue  # imports are neutral
        elif isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.Str)):
            continue  # docstrings are neutral
        elif isinstance(node, ast.Pass):
            continue
        else:
            # Assignments, expressions, if/for/while/with, etc. = executable
            has_executable = True

    if has_definition and not has_executable:
        return True, "Dead code: only function/class definitions without execution"
    return False, ""


# =============================================================================
# LEVEL 1: SYNTAX VALIDATION
# =============================================================================

def validate_syntax(code: str) -> ValidationResult:
    """Level 1: Validate Python syntax using AST parser."""
    try:
        ast.parse(code)
        return ValidationResult(valid=True, level_passed=ValidationLevel.SYNTAX)
    except SyntaxError as e:
        return ValidationResult(
            valid=False, 
            level_passed=ValidationLevel.NONE,
            error=f"SyntaxError at line {e.lineno}: {e.msg}"
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.NONE,
            error=f"Parse error: {str(e)}"
        )


# =============================================================================
# LEVEL 2: EXECUTION VALIDATION
# =============================================================================

EXECUTION_WRAPPER = '''
import sys
import warnings

# Suppress matplotlib GUI
import matplotlib
matplotlib.use('Agg')

# Suppress deprecation warnings (Qiskit 1.2+ has many)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Redirect any plots to prevent blocking
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# Add standard Qiskit imports for test fixture code
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import *
from qiskit.transpiler import Layout, TranspileLayout, PassManager
from qiskit.quantum_info import *
import numpy as np
import re

# Import transpiler passes (try-catch for compatibility)
try:
    from qiskit.transpiler.passes import *
except ImportError:
    pass

# Import primitives (version-compatible)
try:
    from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
    from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
except ImportError:
    try:
        from qiskit.primitives import Sampler, Estimator
    except ImportError:
        pass

# Import fake providers
try:
    from qiskit.providers.fake_provider import *
except ImportError:
    pass

# Import algorithms (optional, not all environments have qiskit_algorithms)
try:
    from qiskit_algorithms import *
    from qiskit_algorithms.minimum_eigensolvers import VQE, QAOA
    from qiskit_algorithms.optimizers import SLSQP, COBYLA, SPSA
    from qiskit_algorithms.utils import algorithm_globals
except ImportError:
    pass

# Import nature (optional)
try:
    from qiskit_nature.second_q.drivers import *
    from qiskit_nature.second_q.mappers import *
    from qiskit_nature.second_q.problems import *
except ImportError:
    pass

# Import Aer (optional)
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import *
except ImportError:
    pass

# Import IBM runtime (optional)
try:
    from qiskit_ibm_runtime import Sampler as RuntimeSampler
    from qiskit_ibm_runtime import Estimator as RuntimeEstimator
    from qiskit_ibm_runtime import Session, Options
except ImportError:
    pass

# Import optimization packages (optional)
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import *
    from qiskit_optimization.converters import *
    from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
    from qiskit_optimization.optimizers import *
    from qiskit_optimization.utils import *
except ImportError:
    pass

# Define common test fixture variables (used in qiskit test code)
qr = QuantumRegister(5, 'qr')
qr1 = QuantumRegister(3, 'qr1')
qr2 = QuantumRegister(3, 'qr2')
qr3 = QuantumRegister(3, 'qr3')
cr = ClassicalRegister(5, 'cr')
cr1 = ClassicalRegister(3, 'cr1')
cr2 = ClassicalRegister(3, 'cr2')
cr_a = ClassicalRegister(3, 'cr_a')
cr_b = ClassicalRegister(3, 'cr_b')
layout_length = 3
bad_arg = "invalid"

# Define mock fixtures for IBM Runtime (if not available)
try:
    service = None
    backend = None
    options = None
except:
    pass

# Execute the code
{code}

# If we reach here, execution succeeded
print("__VALIDATION_SUCCESS__")
'''


def validate_execution(code: str, timeout: int = 30) -> ValidationResult:
    """Level 2: Execute code in sandboxed subprocess."""
    import time
    
    wrapped_code = EXECUTION_WRAPPER.format(code=code)
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.py', 
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(wrapped_code)
        temp_path = Path(f.name)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            timeout=timeout,
            text=True,
            env={
                **dict(__import__('os').environ),
                'MPLBACKEND': 'Agg',
            }
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        if result.returncode != 0:
            # Extract meaningful error message
            stderr = result.stderr.strip()
            # Get last few lines which usually contain the actual error
            error_lines = stderr.split('\n')[-5:]
            error_msg = '\n'.join(error_lines)[:500]
            
            return ValidationResult(
                valid=False,
                level_passed=ValidationLevel.SYNTAX,
                error=f"Runtime error:\n{error_msg}",
                execution_time_ms=execution_time
            )
        
        # Check for success marker
        if "__VALIDATION_SUCCESS__" not in result.stdout:
            return ValidationResult(
                valid=False,
                level_passed=ValidationLevel.SYNTAX,
                error="Execution did not complete normally",
                execution_time_ms=execution_time
            )
        
        # Check for deprecation warnings in stderr
        deprecation_warnings = []
        if result.stderr:
            for line in result.stderr.split('\n'):
                if 'DeprecationWarning' in line or 'deprecated' in line.lower():
                    deprecation_warnings.append(line.strip())
        
        return ValidationResult(
            valid=True,
            level_passed=ValidationLevel.EXECUTION,
            warnings=deprecation_warnings,
            execution_time_ms=execution_time
        )
        
    except subprocess.TimeoutExpired:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.SYNTAX,
            error=f"Execution timeout after {timeout}s"
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.SYNTAX,
            error=f"Execution error: {str(e)}"
        )
    finally:
        temp_path.unlink(missing_ok=True)


# =============================================================================
# LEVEL 3: SEMANTIC VALIDATION
# =============================================================================

SEMANTIC_CHECK_CODE = '''
import sys
import json

# Suppress matplotlib
import matplotlib
matplotlib.use('Agg')

# Execute the user code
{code}

# Perform semantic checks based on category
results = {{"objects_found": [], "checks_passed": [], "checks_failed": []}}

# Check for common Qiskit objects in namespace
local_vars = dict(locals())

# Circuit checks
for name, obj in local_vars.items():
    obj_type = type(obj).__name__
    if obj_type == 'QuantumCircuit':
        results["objects_found"].append({{"name": name, "type": "QuantumCircuit", 
            "num_qubits": obj.num_qubits, "depth": obj.depth()}})
    elif obj_type in ('SamplerResult', 'PrimitiveResult', 'SamplerPubResult'):
        results["objects_found"].append({{"name": name, "type": obj_type}})
    elif obj_type in ('EstimatorResult', 'EstimatorPubResult'):
        results["objects_found"].append({{"name": name, "type": obj_type}})
    elif obj_type == 'SparsePauliOp':
        results["objects_found"].append({{"name": name, "type": "SparsePauliOp",
            "num_qubits": obj.num_qubits}})

# Category-specific checks
category = "{category}"

if category == "circuit_construction":
    circuits = [o for o in results["objects_found"] if o["type"] == "QuantumCircuit"]
    if circuits:
        results["checks_passed"].append("QuantumCircuit created")
        if any(c["depth"] > 0 for c in circuits):
            results["checks_passed"].append("Circuit has gates")
        else:
            results["checks_failed"].append("Circuit is empty")
    else:
        results["checks_failed"].append("No QuantumCircuit found")

elif category == "primitives_api":
    has_result = any(o["type"] in ("SamplerResult", "PrimitiveResult", 
        "SamplerPubResult", "EstimatorResult", "EstimatorPubResult") 
        for o in results["objects_found"])
    if has_result:
        results["checks_passed"].append("Primitive result obtained")
    else:
        results["checks_failed"].append("No primitive result found")

elif category in ("vqe", "qaoa"):
    # Check for optimization result or energy value
    has_observable = any(o["type"] == "SparsePauliOp" for o in results["objects_found"])
    if has_observable:
        results["checks_passed"].append("Hamiltonian/observable created")

else:
    # Universal fallback: check for any Qiskit objects in namespace
    KNOWN_QISKIT_TYPES = {{
        "QuantumCircuit", "SparsePauliOp", "Statevector", "DensityMatrix",
        "Operator", "Clifford", "PassManager", "DAGCircuit",
        "SamplerResult", "PrimitiveResult", "SamplerPubResult",
        "EstimatorResult", "EstimatorPubResult", "BackendV2",
        "AerSimulator", "NoiseModel", "QuantumRegister", "ClassicalRegister",
        "ParameterVector", "Parameter", "TranspileLayout",
    }}
    found_qiskit = False
    for name, obj in local_vars.items():
        if name.startswith("_"):
            continue
        obj_type_name = type(obj).__name__
        obj_module = getattr(type(obj), "__module__", "") or ""
        if obj_type_name in KNOWN_QISKIT_TYPES or "qiskit" in obj_module:
            results["objects_found"].append({{"name": name, "type": obj_type_name}})
            found_qiskit = True
    if found_qiskit:
        results["checks_passed"].append("Qiskit objects found in namespace")
    else:
        results["checks_failed"].append("No Qiskit objects found in namespace")

print("__SEMANTIC_RESULTS__")
print(json.dumps(results))
'''


def validate_semantic(code: str, category: str, timeout: int = 60) -> ValidationResult:
    """Level 3: Category-specific semantic validation."""
    
    check_code = SEMANTIC_CHECK_CODE.format(code=code, category=category)
    
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(check_code)
        temp_path = Path(f.name)
    
    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if "__SEMANTIC_RESULTS__" not in result.stdout:
            return ValidationResult(
                valid=False,
                level_passed=ValidationLevel.EXECUTION,
                error="Semantic check execution failed"
            )
        
        # Parse results
        results_json = result.stdout.split("__SEMANTIC_RESULTS__")[1].strip()
        results = json.loads(results_json)
        
        if results["checks_failed"]:
            return ValidationResult(
                valid=False,
                level_passed=ValidationLevel.EXECUTION,
                error=f"Semantic checks failed: {', '.join(results['checks_failed'])}"
            )
        
        return ValidationResult(
            valid=True,
            level_passed=ValidationLevel.SEMANTIC,
            warnings=[f"Found: {o['type']}" for o in results["objects_found"]]
        )
        
    except subprocess.TimeoutExpired:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.EXECUTION,
            error=f"Semantic validation timeout after {timeout}s"
        )
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.EXECUTION,
            error=f"Failed to parse semantic results: {e}"
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.EXECUTION,
            error=f"Semantic validation error: {str(e)}"
        )
    finally:
        temp_path.unlink(missing_ok=True)


# =============================================================================
# LEVEL 3b: TEST-CASE-BASED VALIDATION
# =============================================================================

TEST_CASE_CODE = '''
import sys
import warnings

# Suppress matplotlib GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# Execute the user code (must include its own imports)
{code}

# Run test assertions
{test_code}

print("__TEST_PASSED__")
'''


def validate_test_code(code: str, test_code: str, timeout: int = 60) -> ValidationResult:
    """Level 3: Test-case-based semantic validation.

    Runs the model's code followed by assertion-based test code.
    The model's code must be self-contained (include its own imports).
    """
    import time

    wrapped = TEST_CASE_CODE.format(code=code, test_code=test_code)

    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(wrapped)
        temp_path = Path(f.name)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            timeout=timeout,
            text=True,
            env={
                **dict(__import__('os').environ),
                'MPLBACKEND': 'Agg',
            }
        )

        execution_time = (time.time() - start_time) * 1000

        if result.returncode != 0:
            stderr = result.stderr.strip()
            error_lines = stderr.split('\n')[-5:]
            error_msg = '\n'.join(error_lines)[:500]

            # Distinguish assertion failures from runtime errors
            if 'AssertionError' in stderr:
                return ValidationResult(
                    valid=False,
                    level_passed=ValidationLevel.EXECUTION,
                    error=f"Test failed: {error_msg}",
                    execution_time_ms=execution_time
                )

            return ValidationResult(
                valid=False,
                level_passed=ValidationLevel.SYNTAX,
                error=f"Runtime error:\n{error_msg}",
                execution_time_ms=execution_time
            )

        if "__TEST_PASSED__" not in result.stdout:
            return ValidationResult(
                valid=False,
                level_passed=ValidationLevel.EXECUTION,
                error="Test execution did not complete",
                execution_time_ms=execution_time
            )

        return ValidationResult(
            valid=True,
            level_passed=ValidationLevel.SEMANTIC,
            execution_time_ms=execution_time
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.EXECUTION,
            error=f"Test validation timeout after {timeout}s"
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.EXECUTION,
            error=f"Test validation error: {str(e)}"
        )
    finally:
        temp_path.unlink(missing_ok=True)


# =============================================================================
# CODE EXTRACTION
# =============================================================================

def extract_code(response: str) -> Optional[str]:
    """Extract Python code block from model response."""
    # Try to find ```python ... ``` block
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return code_part.strip()
    
    # Try generic ``` block
    if "```" in response:
        parts = response.split("```")
        if len(parts) >= 2:
            # Take first code block
            code_part = parts[1]
            # Remove language identifier if present
            lines = code_part.strip().split('\n')
            if lines and lines[0].strip() in ('python', 'py', ''):
                lines = lines[1:]
            return '\n'.join(lines).strip()
    
    # If no code blocks, check if the entire response looks like code
    if response.strip().startswith(('import ', 'from ', '#')):
        return response.strip()
    
    return None


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def validate_example(
    example: dict,
    run_semantic: bool = True,
    execution_timeout: int = 30,
    semantic_timeout: int = 60
) -> tuple[ValidationResult, Optional[str]]:
    """
    Run full validation pipeline on a generated example.
    
    Returns (ValidationResult, extracted_code)
    """
    response = example.get("response", "")
    category = example.get("category", example.get("metadata", {}).get("category", "unknown"))
    
    # Extract code
    code = extract_code(response)
    if not code:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.NONE,
            error="No code block found in response"
        ), None
    
    # Check for deprecated patterns
    deprecated_issues = check_deprecated_patterns(code)
    if deprecated_issues:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.NONE,
            error=f"Deprecated patterns found: {'; '.join(deprecated_issues)}"
        ), code
    
    # Check for required patterns
    required_issues = check_required_patterns(code, category)
    # Required patterns are warnings, not failures
    
    # Level 1: Syntax
    result = validate_syntax(code)
    if not result.valid:
        return result, code

    # Check for test stubs (between syntax and execution)
    is_stub, stub_reason = detect_test_stubs(code)
    if is_stub:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.SYNTAX,
            error=stub_reason
        ), code

    # Check for dead code (between syntax and execution)
    is_dead, dead_reason = detect_dead_code(code)
    if is_dead:
        return ValidationResult(
            valid=False,
            level_passed=ValidationLevel.SYNTAX,
            error=dead_reason
        ), code

    # Level 2: Execution
    result = validate_execution(code, timeout=execution_timeout)
    if not result.valid:
        return result, code
    
    # Level 3: Semantic (optional)
    if run_semantic:
        # Prefer test-case-based validation when test_code is available
        test_code = example.get("test_code")
        if test_code:
            result = validate_test_code(code, test_code, timeout=semantic_timeout)
        else:
            result = validate_semantic(code, category, timeout=semantic_timeout)
        result.warnings = (result.warnings or []) + required_issues
        return result, code
    
    result.warnings = (result.warnings or []) + required_issues
    return result, code


