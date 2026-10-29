from pathlib import Path


root = Path(SPECPATH)

a = Analysis(
    [str(root / "web_launcher.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "src" / "service" / "index.html"), "src/service"),
        (str(root / "config.yaml"), "."),
        (str(root / ".env.example"), "."),
        (str(root / "README.md"), "."),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "IPython",
        "black",
        "docutils",
        "jupyter",
        "matplotlib",
        "nbformat",
        "notebook",
        "pytest",
        "sphinx",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TraceDocRAG",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TraceDocRAG",
)
