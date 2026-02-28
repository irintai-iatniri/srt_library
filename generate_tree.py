import os

def generate_recursive_tree(path, exclude_dirs, exclude_exts, indent=""):
    items = sorted(os.listdir(path))
    res = ""
    
    # Filter items
    filtered_items = []
    for item in items:
        if item.startswith('.') or item in exclude_dirs or any(item.endswith(ext) for ext in exclude_exts):
            continue
        filtered_items.append(item)
    
    for i, item in enumerate(filtered_items):
        is_last = (i == len(filtered_items) - 1)
        prefix = "└── " if is_last else "├── "
        child_indent = "    " if is_last else "│   "
        
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            res += f"{indent}{prefix}{item}/\n"
            res += generate_recursive_tree(item_path, exclude_dirs, exclude_exts, indent + child_indent)
        else:
            res += f"{indent}{prefix}{item}\n"
    return res

if __name__ == "__main__":
    path = "/media/Andrew/Backup/Programs/srt_library"
    exclude_dirs = {
        '__pycache__', '.git', '.venv', '.claude', 'target', 
        '_core.libs', '_libs', 'node_modules', 'topology_images',
        'build', 'dist', '.pytest_cache', '.ipynb_checkpoints'
    }
    exclude_exts = {
        '.so', '.pyc', '.pyo', '.ptx', '.pdf', '.txt', '.so.1', '.so.13', '.so.0', '.so.5',
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.wav', '.mp3', '.mp4'
    }
    
    output = "# srt_library — Directory Tree\n\n"
    output += f"*Last updated: {os.popen('date +%Y-%m-%d').read().strip()}*\n\n"
    output += "```\n"
    output += "srt_library/\n"
    output += generate_recursive_tree(path, exclude_dirs, exclude_exts)
    output += "```\n"
    
    with open("tree.md", "w") as f:
        f.write(output)
    print("tree.md regenerated with deep recursion.")
