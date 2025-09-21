import re
import sys
from pathlib import Path

CONFIG_PATH = Path('app/config.py')

VERSION_RE = re.compile(r'MODEL_VERSION\s*:\s*str\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"')

def bump(part: str):
    text = CONFIG_PATH.read_text(encoding='utf-8')
    m = VERSION_RE.search(text)
    if not m:
        print('Current version not found', file=sys.stderr)
        sys.exit(1)
    major, minor, patch = map(int, m.group(1).split('.'))
    if part == 'major':
        major += 1; minor = 0; patch = 0
    elif part == 'minor':
        minor += 1; patch = 0
    elif part == 'patch':
        patch += 1
    else:
        print('Part must be one of: major, minor, patch', file=sys.stderr)
        sys.exit(1)
    new_version = f'{major}.{minor}.{patch}'
    new_text = VERSION_RE.sub(f'MODEL_VERSION: str = "{new_version}"', text)
    CONFIG_PATH.write_text(new_text, encoding='utf-8')
    print(new_version)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python scripts/bump_version.py [major|minor|patch]')
        sys.exit(1)
    bump(sys.argv[1])
