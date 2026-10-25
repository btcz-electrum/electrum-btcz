# -*- mode: python -*-
# Windows build on GitHub Actions (.github/workflows/build.yml): one-file BTCZ-Electrum-<version>.exe
import os
import os.path
import sys
from PyInstaller.utils.hooks import (collect_data_files, collect_submodules,
                                     collect_dynamic_libs)


for i, x in enumerate(sys.argv):
    if x == '--name':
        cmdline_name = sys.argv[i+1]
        break
else:
    cmdline_name = 'BTCZ-Electrum-%s' % os.environ.get('ZCASH_ELECTRUM_VERSION', '')

GITHUB_REF = os.environ.get('GITHUB_REF')
ZCASH_ELECTRUM_VERSION = os.environ.get('ZCASH_ELECTRUM_VERSION')

hiddenimports = []
hiddenimports += collect_submodules('pkg_resources')  # workaround for https://github.com/pypa/setuptools/issues/1963
hiddenimports += collect_submodules('trezorlib')
hiddenimports += collect_submodules('safetlib')
hiddenimports += collect_submodules('btchip')
hiddenimports += collect_submodules('keepkeylib')
hiddenimports += collect_submodules('websocket')

# safetlib imports PyQt5.Qt.  We use a local updated copy of pinmatrix.py until they
# release a new version that includes https://github.com/archos-safe-t/python-safet/commit/b1eab3dba4c04fdfc1fcf17b66662c28c5f2380e
if 'safetlib.qt.pinmatrix' in hiddenimports: hiddenimports.remove('safetlib.qt.pinmatrix')

hiddenimports += [
    'electrum_btcz',
    'electrum_btcz.base_crash_reporter',
    'electrum_btcz.base_wizard',
    'electrum_btcz.plot',
    'electrum_btcz.qrscanner',
    'electrum_btcz.websockets',
    'electrum_btcz.gui.qt',
    'PyQt5.sip',
    'PyQt5.QtPrintSupport',  # needed by Revealer

    'electrum_btcz.plugins',

    'electrum_btcz.plugins.hw_wallet.qt',

    'electrum_btcz.plugins.audio_modem.qt',
    'electrum_btcz.plugins.cosigner_pool.qt',
    'electrum_btcz.plugins.digitalbitbox.qt',
    'electrum_btcz.plugins.email_requests.qt',
    'electrum_btcz.plugins.keepkey.qt',
    'electrum_btcz.plugins.revealer.qt',
    'electrum_btcz.plugins.labels.qt',
    'electrum_btcz.plugins.scan_over_gap.qt',
    'electrum_btcz.plugins.trezor.qt',
    'electrum_btcz.plugins.safe_t.client',
    'electrum_btcz.plugins.safe_t.qt',
    'electrum_btcz.plugins.ledger.qt',
    'electrum_btcz.plugins.virtualkeyboard.qt',
]

datas = [
    ('electrum_btcz/checkpoints*.*', 'electrum_btcz'),
    ('electrum_btcz/*.json', 'electrum_btcz'),
    ('electrum_btcz/locale', 'electrum_btcz/locale'),
    ('electrum_btcz/wordlist', 'electrum_btcz/wordlist'),
    ('electrum_btcz/gui/icons', 'electrum_btcz/gui/icons'),
]

datas += collect_data_files('trezorlib')
datas += collect_data_files('safetlib')
datas += collect_data_files('btchip')
datas += collect_data_files('keepkeylib')

binaries = [('electrum_btcz/libsecp256k1-0.dll', 'electrum_btcz')]

# Workaround for "Retro Look":
binaries += [b for b in collect_dynamic_libs('PyQt5') if 'qwindowsvista' in b[0]]

# https://github.com/pyinstaller/pyinstaller/wiki/Recipe-remove-tkinter-tcl
sys.modules['FixTk'] = None
excludes = ['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter']
# the Android (kivy) GUI exits on import when kivy is missing
excludes += ['electrum_btcz.gui.kivy', 'kivy']
excludes += [
    'PyQt5.QtBluetooth',
    'PyQt5.QtCLucene',
    'PyQt5.QtDBus',
    'PyQt5.Qt5CLucene',
    'PyQt5.QtDesigner',
    'PyQt5.QtDesignerComponents',
    'PyQt5.QtHelp',
    'PyQt5.QtLocation',
    'PyQt5.QtMultimedia',
    'PyQt5.QtMultimediaQuick_p',
    'PyQt5.QtMultimediaWidgets',
    'PyQt5.QtNetwork',
    'PyQt5.QtNetworkAuth',
    'PyQt5.QtNfc',
    'PyQt5.QtOpenGL',
    'PyQt5.QtPositioning',
    'PyQt5.QtQml',
    'PyQt5.QtQuick',
    'PyQt5.QtQuickParticles',
    'PyQt5.QtQuickWidgets',
    'PyQt5.QtSensors',
    'PyQt5.QtSerialPort',
    'PyQt5.QtSql',
    'PyQt5.Qt5Sql',
    'PyQt5.Qt5Svg',
    'PyQt5.QtTest',
    'PyQt5.QtWebChannel',
    'PyQt5.QtWebEngine',
    'PyQt5.QtWebEngineCore',
    'PyQt5.QtWebEngineWidgets',
    'PyQt5.QtWebKit',
    'PyQt5.QtWebKitWidgets',
    'PyQt5.QtWebSockets',
    'PyQt5.QtXml',
    'PyQt5.QtXmlPatterns',
    'PyQt5.QtWebProcess',
]

a = Analysis(['electrum-btcz'],
             hiddenimports=hiddenimports,
             datas=datas,
             binaries=binaries,
             excludes=excludes,
             runtime_hooks=['pyi_runtimehook.py'])

# http://stackoverflow.com/questions/19055089/
for d in a.datas:
    if 'pyconfig' in d[0]:
        a.datas.remove(d)
        break

# Strip out parts of Qt that we never use. Reduces binary size by tens of MBs. see #4815
qt_bins2remove=('qt5web', 'qt53d', 'qt5game', 'qt5designer', 'qt5quick', 'qt5location', 'qt5test', 'qt5xml')
print("Removing Qt binaries:", *qt_bins2remove)
for x in a.binaries.copy():
    for r in qt_bins2remove:
        if x[0].lower().startswith(r):
            a.binaries.remove(x)
            print('----> Removed x =', x)

pyz = PYZ(a.pure)

# single portable .exe; wallets are kept in %APPDATA%\Electrum-BTCZ as usual
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.datas,
          debug=False,
          strip=False,
          upx=False,
          console=False,
          icon='electrum_btcz/gui/icons/electrum-btcz.ico',
          name=os.path.join('dist', cmdline_name))
