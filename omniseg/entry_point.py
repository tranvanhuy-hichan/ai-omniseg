"""PyInstaller entry point. Kept separate from __main__.py so a frozen build
has a single importable script instead of relying on `python -m omniseg`.
"""
from omniseg.app import main

if __name__ == '__main__':
    main()
