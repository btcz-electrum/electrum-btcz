BitcoinZ Electrum - lightweight BitcoinZ wallet
===============================================

::

  Licence: MIT Licence
  Language: Python (>= 3.8)
  Homepage: https://github.com/btcz-electrum/electrum-btcz
  Releases: https://github.com/btcz-electrum/electrum-btcz/releases

BitcoinZ Electrum is a fast desktop wallet for BitcoinZ (BTCZ). It does not download the
blockchain; it talks to BitcoinZ Electrum servers (electrum1, electrum2 and electrum6.btcz.rocks)
and keeps your keys on your own computer.

**Current version: 4.1.2.4.** It is a port of Electrum-Zcash 4.1.2 to BitcoinZ and replaces the
old 3.1.3 wallet.

.. important::

   **The old 3.1.3 wallet can no longer send BTCZ.** BitcoinZ activated the Canopy upgrade at
   block 1,735,000 and 3.1.3 signs transactions with the old (Sapling) rules, so the network
   rejects them. Please update to 4.1.2.4. Your existing wallets keep working, see
   `Upgrading from 3.1.3`_.


Download and install
====================

Download the files from the latest release:
https://github.com/btcz-electrum/electrum-btcz/releases/latest

=====================================================  ==============================================
File                                                   For
=====================================================  ==============================================
``BTCZ-Electrum-4.1.2.4-setup.exe``                    Windows 10/11 (64-bit), installer (recommended)
``BTCZ-Electrum-4.1.2.4.exe``                          Windows, portable single file, no installation
``BTCZ-Electrum-4.1.2.4-macos-arm64.dmg``              macOS 11+ on Apple Silicon (M1, M2, M3, M4)
``BTCZ-Electrum-4.1.2.4-macos-intel.dmg``              macOS on Intel Macs
``btcz_electrum-4.1.2.4.tar.gz``                       Linux and other systems (source)
``SHA256SUMS.txt``                                     checksums of all files
=====================================================  ==============================================

Windows
-------

1. Run ``BTCZ-Electrum-4.1.2.4-setup.exe``. The files are not code-signed yet, so Windows
   SmartScreen may show "Windows protected your PC": click **More info**, then **Run anyway**.
2. The setup first removes any older BTCZ Electrum it finds (also the 3.x installation), then
   installs into ``C:\Program Files\BitcoinZ Electrum`` with a desktop and Start menu shortcut.
3. Start **BitcoinZ Electrum**. Your wallets are not touched by the setup or by the uninstaller.

The portable ``BTCZ-Electrum-4.1.2.4.exe`` needs no installation: just run it. It uses the same
wallet folder as the installed version. Old portable ``.exe`` files can simply be deleted.

macOS
-----

1. Open the ``.dmg`` for your Mac (Apple menu > About This Mac shows "Apple M…" or "Intel")
   and drag **BitcoinZ Electrum** to Applications.
2. The app is not notarized yet: the first time, right-click it in Applications and choose
   **Open**, then **Open** again (or allow it in System Settings > Privacy & Security).

Linux
-----

Install the dependencies, build libsecp256k1 and run the wallet from the unpacked source::

    sudo apt-get install python3-pip python3-pyqt5 git automake libtool build-essential
    tar xzf btcz_electrum-4.1.2.4.tar.gz && cd btcz_electrum-4.1.2.4
    python3 -m pip install --user .[fast]
    ./contrib/make_libsecp256k1.sh       # or: sudo apt-get install libsecp256k1-dev
    ./electrum-btcz

Checking the download
---------------------

Compare the SHA-256 of the file with the line in ``SHA256SUMS.txt``:

- Windows (PowerShell): ``Get-FileHash .\BTCZ-Electrum-4.1.2.4-setup.exe``
- macOS / Linux: ``shasum -a 256 BTCZ-Electrum-4.1.2.4-macos-arm64.dmg``


Upgrading from 3.1.3
====================

1. **Before anything else, make sure you have your seed words written down** (old wallet:
   Wallet > Seed), or make a copy of your wallet folder.
2. Install 4.1.2.4 as described above and start it. It opens your existing wallet.
3. The first time the wallet is opened, 4.1.2.4 converts the wallet file to the new format.
   After that the file can no longer be opened by 3.1.3.

Where the wallets are stored:

- **Windows:** ``%APPDATA%\Electrum-BTCZ\wallets``. Older versions used
  ``%APPDATA%\Electrum-ZCASH``. When ``Electrum-BTCZ`` does not exist yet, 4.1.2.4 creates it
  and copies the wallet files from ``Electrum-ZCASH\wallets`` into it; the old folder stays as
  it was, as a backup. If ``Electrum-BTCZ`` already exists, only that folder is used.
- **macOS and Linux:** ``~/.electrum-btcz/wallets`` (unchanged).

A wallet stored elsewhere can be opened with File > Open, or restored from its seed with
File > New/Restore.


What is new in 4.1.2.x
======================

- Based on Electrum-Zcash 4.1.2 (previously Electrum 3.1.3): newer Python and Qt, faster
  synchronisation, better fee handling, hardware wallet libraries updated.
- Transactions are signed with the Canopy rules, so sending works again.
- Checkpoints up to block 1,861,199; default servers electrum1, electrum2 and
  electrum6.btcz.rocks.
- New look: BitcoinZ branding instead of Zcash, balance header with fiat value, flat light,
  dark and "follow system" themes (Tools > Preferences > General > Color theme).
- Windows installer that replaces older versions; wallets in ``%APPDATA%\Electrum-BTCZ``.
- The "use Tor?" question is no longer shown on every start (Tor can still be set up in
  Network preferences).
- Optional development fee, see below.


Development fee
===============

When you send BTCZ, the Confirm Transaction dialog adds an **optional** development fee of
**0.1%** of the amount sent, paid to ``t1b3Q9vSWzJsGsayHTi9ivcrNCk8xR2Nt1M``:

- no fee for amounts below 100,000 BTCZ, at most 50,000 BTCZ per transaction;
- no fee when you send to your own addresses;
- the fee is shown as its own line before you confirm;
- it can be switched off with the checkbox "Support BTCZ Electrum development (0.1%)" in
  the same dialog, and the wallet remembers your choice.

Thank you to everyone who leaves it on: it helps keep the wallet and the Electrum servers
running.


Getting help
============

Report problems at https://github.com/btcz-electrum/electrum-btcz/issues. Never share your
seed words or wallet file with anyone, including people offering help.


Development version
===================

Check out the code from GitHub::

    git clone -b btcz https://github.com/btcz-electrum/electrum-btcz.git
    cd electrum-btcz

Run install (this should install dependencies)::

    python3 -m pip install .[fast]


Compile the protobuf description file::

    sudo apt-get install protobuf-compiler
    protoc --proto_path=electrum_btcz --python_out=electrum_btcz electrum_btcz/paymentrequest.proto

Create translations (optional)::

    sudo apt-get install python-requests gettext
    ./contrib/make_locale

Builds
------

``.github/workflows/build.yml`` builds the Windows setup and portable ``.exe`` (``win.spec``,
``contrib/github/btcz-electrum-setup.nsi``), both macOS ``.dmg`` files (``osx.spec``) and the
source ``.tar.gz`` on GitHub Actions. Pushing a tag ``v<version>`` creates a release with all
files and ``SHA256SUMS.txt``.

The previous 3.1.3 code is kept in the ``btcz-3.1.3`` branch.
