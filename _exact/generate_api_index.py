#!/usr/bin/env python3
"""
Generate API_INDEX.md - Maps Python function names to Rust source locations.

Parses Rust files for #[pyfunction], #[pymethods], and #[pyclass] attributes
to create a comprehensive index for AI navigation.
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


def parse_rust_file(file_path: Path) -> List[Dict[str, any]]:
    """Parse a Rust file for Python-visible items."""
    items = []

    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return items

    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # Match #[pyfunction]
        if '#[pyfunction]' in line or '#[pyfunction(' in line:
            # Look ahead for the function name
            for j in range(i, min(i + 5, len(lines) + 1)):
                fn_match = re.search(r'(?:pub\s+)?fn\s+(\w+)', lines[j-1])
                if fn_match:
                    items.append({
                        'type': 'function',
                        'name': fn_match.group(1),
                        'file': str(file_path),
                        'line': j,
                        'decorator': line.strip()
                    })
                    break

        # Match #[pyclass]
        if '#[pyclass]' in line or '#[pyclass(' in line:
            # Look ahead for the struct/enum name
            for j in range(i, min(i + 5, len(lines) + 1)):
                class_match = re.search(r'(?:pub\s+)?(?:struct|enum)\s+(\w+)', lines[j-1])
                if class_match:
                    items.append({
                        'type': 'class',
                        'name': class_match.group(1),
                        'file': str(file_path),
                        'line': j,
                        'decorator': line.strip()
                    })
                    break

        # Match #[pymethods]
        if '#[pymethods]' in line:
            # Look ahead for impl block
            for j in range(i, min(i + 5, len(lines) + 1)):
                impl_match = re.search(r'impl\s+(\w+)', lines[j-1])
                if impl_match:
                    class_name = impl_match.group(1)
                    # Parse methods in this impl block
                    methods = parse_pymethods_block(lines, j, class_name, file_path)
                    items.extend(methods)
                    break

    return items


def parse_pymethods_block(lines: List[str], start_line: int, class_name: str, file_path: Path) -> List[Dict]:
    """Parse methods within a #[pymethods] impl block."""
    methods = []
    brace_depth = 0
    in_block = False

    for i in range(start_line, len(lines)):
        line = lines[i]

        # Track braces to know when we exit the impl block
        if '{' in line:
            brace_depth += line.count('{')
            in_block = True
        if '}' in line:
            brace_depth -= line.count('}')
            if brace_depth == 0 and in_block:
                break

        # Look for method definitions
        fn_match = re.search(r'(?:pub\s+)?fn\s+(\w+)', line)
        if fn_match and in_block:
            method_name = fn_match.group(1)
            # Skip internal Rust methods like __new__, __repr__, etc. if they don't have special Python names
            methods.append({
                'type': 'method',
                'name': f"{class_name}.{method_name}",
                'class': class_name,
                'method': method_name,
                'file': str(file_path),
                'line': i + 1
            })

    return methods


def find_rust_files(base_dir: Path) -> List[Path]:
    """Find all .rs files in the directory."""
    return list(base_dir.rglob('*.rs'))


def generate_index(precision_mode: str, rust_dir: Path) -> List[Dict]:
    """Generate index for a specific precision mode."""
    all_items = []

    rust_files = find_rust_files(rust_dir)
    print(f"  Scanning {len(rust_files)} Rust files in {precision_mode}...")

    for rust_file in rust_files:
        items = parse_rust_file(rust_file)
        for item in items:
            # Make file path relative to srt_library root
            rel_path = rust_file.relative_to(rust_dir.parent.parent)
            item['file'] = str(rel_path)
            item['precision'] = precision_mode
        all_items.extend(items)

    return all_items


def format_markdown_table(items: List[Dict]) -> str:
    """Format items as a markdown table."""
    # Group by type
    by_type = defaultdict(list)
    for item in items:
        by_type[item['type']].append(item)

    output = []

    # Functions
    if by_type['function']:
        output.append("### Functions\n")
        output.append("| Python Function | Precision | Rust File | Line |")
        output.append("|----------------|-----------|-----------|------|")

        for item in sorted(by_type['function'], key=lambda x: x['name']):
            output.append(f"| `{item['name']}()` | {item['precision']} | `{item['file']}` | {item['line']} |")
        output.append("")

    # Classes
    if by_type['class']:
        output.append("### Classes\n")
        output.append("| Python Class | Precision | Rust File | Line |")
        output.append("|-------------|-----------|-----------|------|")

        for item in sorted(by_type['class'], key=lambda x: x['name']):
            output.append(f"| `{item['name']}` | {item['precision']} | `{item['file']}` | {item['line']} |")
        output.append("")

    # Methods
    if by_type['method']:
        output.append("### Methods\n")
        output.append("| Python Method | Class | Precision | Rust File | Line |")
        output.append("|--------------|-------|-----------|-----------|------|")

        for item in sorted(by_type['method'], key=lambda x: x['name']):
            output.append(f"| `{item['name']}()` | `{item['class']}` | {item['precision']} | `{item['file']}` | {item['line']} |")
        output.append("")

    return '\n'.join(output)


def main():
    """Main entry point."""
    srt_library = Path(__file__).parent

    print("=" * 80)
    print("Generating API_INDEX.md")
    print("=" * 80)

    # Parse both precision modes
    exact_items = []
    float_items = []

    exact_dir = srt_library / 'exact_arithmetic' / 'rust' / 'src'
    if exact_dir.exists():
        print(f"\nScanning exact_arithmetic...")
        exact_items = generate_index('exact', exact_dir)
        print(f"  Found {len(exact_items)} items")

    float_dir = srt_library / 'float_arithmetic' / 'rust' / 'src'
    if float_dir.exists():
        print(f"\nScanning float_arithmetic...")
        float_items = generate_index('float', float_dir)
        print(f"  Found {len(float_items)} items")

    all_items = exact_items + float_items

    if not all_items:
        print("\nError: No Python-visible items found!")
        return

    # Generate markdown
    print(f"\nGenerating markdown table...")

    output = [
        "# SRT Library API Index",
        "",
        f"**Generated**: Automated scan of Rust source files",
        f"**Total items**: {len(all_items)}",
        "",
        "This index maps Python-visible functions, classes, and methods to their Rust implementations.",
        "",
        "## Quick Navigation",
        "",
        "- [Functions](#functions) - Standalone functions exposed to Python",
        "- [Classes](#classes) - Python classes implemented in Rust",
        "- [Methods](#methods) - Methods on Python classes",
        "",
        "## Precision Modes",
        "",
        "- **exact**: `exact_arithmetic/` - Exact arithmetic (no floats)",
        "- **float**: `float_arithmetic/` - Float arithmetic (traditional)",
        "",
        "---",
        "",
        format_markdown_table(all_items),
        "---",
        "",
        "## Usage",
        "",
        "1. **Find a Python function**: Search this page for the function name",
        "2. **Navigate to source**: Open the file at the specified line",
        "3. **Choose precision**: Use `exact` for theoretical work, `float` for practical applications",
        "",
        "## Regenerating This Index",
        "",
        "```bash",
        "cd srt_library",
        "python generate_api_index.py > API_INDEX.md",
        "```",
    ]

    output_file = srt_library / 'API_INDEX.md'
    output_file.write_text('\n'.join(output), encoding='utf-8')

    print(f"\n✅ Generated {output_file}")
    print(f"   Total items indexed: {len(all_items)}")
    print(f"   - Functions: {len([i for i in all_items if i['type'] == 'function'])}")
    print(f"   - Classes: {len([i for i in all_items if i['type'] == 'class'])}")
    print(f"   - Methods: {len([i for i in all_items if i['type'] == 'method'])}")
    print("=" * 80)


if __name__ == '__main__':
    main()
