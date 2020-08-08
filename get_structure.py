import os

def list_files(startpath):
    with open('structure.txt', 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = '│   ' * level
            f.write(f'{indent}├── {os.path.basename(root)}/\n')
            subindent = '│   ' * (level + 1)
            for file in files:
                f.write(f'{subindent}├── {file}\n')

# Joriy papkada ishlatish uchun
list_files('.')