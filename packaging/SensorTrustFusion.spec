# PyInstaller specification of the Windows executable of SensorTrust Fusion v1.0.0.
# Build (Windows, from the repository root):
#     pip install -r requirements.txt pyinstaller
#     pip install -e .
#     pyinstaller packaging/SensorTrustFusion.spec --noconfirm
# Result: dist/SensorTrustFusion.exe (single file, no Python installation required).
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
hidden = collect_submodules("sensortrust") + collect_submodules("SALib") + [
    "scipy.stats._qmc", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf",
]

a = Analysis(
    [os.path.join(SPECPATH, "sensortrust_app.py")],
    pathex=[os.path.join(SPECPATH, "..", "src")],
    binaries=[],
    # data files loaded at run time (e.g. scipy/stats/_sobol_direction_numbers.npz for Sobol sequences)
    datas=collect_data_files("scipy.stats") + collect_data_files("SALib")
    + [(os.path.join(SPECPATH, "..", "Manuales", f), "Manuales") for f in
       ("Manual_de_Usuario_SensorTrust_Fusion.docx", "Manual_Tecnico_SensorTrust_Fusion.docx")
       if os.path.exists(os.path.join(SPECPATH, "..", "Manuales", f))],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["IPython", "jupyter", "notebook", "PySide6.QtWebEngineCore",
              "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml",
              "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
# start-up image shown while the single-file executable unpacks (closed when the main window opens)
splash = Splash(os.path.join(SPECPATH, "splash.png"), binaries=a.binaries, datas=a.datas,
                text_pos=None, always_on_top=False)
exe = EXE(
    pyz, a.scripts, splash, splash.binaries, a.binaries, a.zipfiles, a.datas, [],
    name="SensorTrustFusion",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    version=os.path.join(SPECPATH, "version_info.txt"),
)
