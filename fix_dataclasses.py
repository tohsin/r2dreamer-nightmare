import os
import glob
import re

count = 0
for filepath in glob.glob('/home/tosin/Documents/GitHub/r2dreamer-nightmare/src/safety-gymnasium/safety-gymnasium-main/**/*.py', recursive=True):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if re.search(r':\s*np\.(?:array|ndarray)\s*=\s*(?:COLOR|GROUP)\[.*?\]', content):
        if 'import dataclasses\n' not in content:
            content = 'import dataclasses\n' + content
            
        content, n1 = re.subn(r'(:\s*np\.(?:array|ndarray)\s*=\s*)(COLOR\[.*?\])', r'\1dataclasses.field(default_factory=lambda: \2)', content)
        content, n2 = re.subn(r'(:\s*np\.(?:array|ndarray)\s*=\s*)(GROUP\[.*?\])', r'\1dataclasses.field(default_factory=lambda: \2)', content)
        
        if n1 > 0 or n2 > 0:
            count += 1
            with open(filepath, 'w') as f:
                f.write(content)
                
print(f"Fixed {count} files.")
