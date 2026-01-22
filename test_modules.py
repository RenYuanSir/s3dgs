"""
Test script to verify the refactored module structure.
This script checks if all files exist and have correct syntax.
"""

import os
import ast

def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"[OK] {filepath} exists")
        return True
    else:
        print(f"[FAIL] {filepath} not found")
        return False

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"[OK] {filepath} has valid syntax")
        return True
    except SyntaxError as e:
        print(f"[FAIL] {filepath} has syntax error: {e}")
        return False

def check_function_exists(filepath, items):
    """Check if specific functions or classes exist in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)

        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        missing = []
        for name in items:
            if name not in functions and name not in classes:
                missing.append(name)

        if missing:
            print(f"[FAIL] {filepath} missing items: {missing}")
            return False
        else:
            print(f"[OK] {filepath} contains all required functions and classes")
            return True
    except Exception as e:
        print(f"[FAIL] Error checking items in {filepath}: {e}")
        return False

print("="*60)
print("S-3DGS Module Structure Verification")
print("="*60)

base_dir = "s3dgs"

# Check module structure
modules = {
    "loss.py": ["l1_loss", "scale_invariant_depth_loss", "semantic_loss_with_gating"],
    "render.py": ["render_dual_pass", "_render_depth_map", "_render_semantic_map"],
    "model.py": ["SemanticGaussianModel"],
    "train.py": ["train"],
    "dataset.py": ["TomatoDataset", "create_dataloader"]
}

all_ok = True

for filename, items in modules.items():
    filepath = os.path.join(base_dir, filename)
    print(f"\nChecking {filename}:")
    all_ok &= check_file_exists(filepath)
    all_ok &= check_python_syntax(filepath)
    if items:
        all_ok &= check_function_exists(filepath, items)

print("\n" + "="*60)
if all_ok:
    print("SUCCESS: All modules are properly structured!")
else:
    print("FAILURE: Some modules have issues")
print("="*60)

print("\nModule structure:")
print("  s3dgs/")
print("    ├── __init__.py")
print("    ├── model.py      - SemanticGaussianModel")
print("    ├── dataset.py    - TomatoDataset, create_dataloader")
print("    ├── loss.py       - l1_loss, scale_invariant_depth_loss, semantic_loss_with_gating")
print("    ├── render.py     - render_dual_pass, _render_depth_map, _render_semantic_map")
print("    └── train.py      - Main training loop (refactored)")
