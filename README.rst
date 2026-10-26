BitcoinZ Electrum - Lightweight BitcoinZ client
=====================================

**Version 4.1.2.4 (testing)** - port of Electrum-Zcash 4.1.2 to BitcoinZ.

- Signs transactions with the Canopy rules (active on BitcoinZ since block 1,735,000);
  the old 3.1.3 client can no longer send.
- Servers: electrum1/electrum2/electrum6.btcz.rocks; checkpoints up to block 1,861,199.
- Builds: Windows ``.exe``, macOS ``.dmg`` (Apple Silicon and Intel) and source ``.tar.gz``
  are built by GitHub Actions (``.github/workflows/build.yml``) and attached to Releases.
- Opening a 3.1.3 wallet upgrades the wallet file; back it up first.
- Windows: ``BTCZ-Electrum-<version>-setup.exe`` installs to Program Files and first removes
  any older installed BTCZ Electrum (also the 3.x one); ``BTCZ-Electrum-<version>.exe`` is the
  portable single file. Neither the setup nor the uninstaller touches the wallets.
- Windows wallets are in ``%APPDATA%\\Electrum-BTCZ`` (older versions used ``Electrum-ZCASH``).
  When ``Electrum-BTCZ`` does not exist yet it is created and the wallet files from
  ``Electrum-ZCASH\\wallets`` are copied into it; the old folder is left unchanged.

Development fee
---------------

The Confirm Transaction dialog adds an optional development fee of 0.1% of the amount sent
(none below 100,000 BTCZ, at most 50,000 BTCZ, none on sends to your own addresses),
paid to ``t1b3Q9vSWzJsGsayHTi9ivcrNCk8xR2Nt1M``. It is shown as its own line and can be
switched off with the checkbox in that dialog; the choice is remembered.


::

  Licence: MIT Licence
  Author: Thomas Voegtlin
  Language: Python (>= 3.6)
  Homepage: https://github.com/btcz-electrum/electrum-btcz






Getting started
===============


Use PPA setup
-------------

On Ubuntu/Linux Mint you can try to install BitcoinZ Electrum with next commands::

    sudo add-apt-repository ppa:zebra-lucky/btcz-electrum
    sudo apt-get update
    sudo apt-get install btcz-electrum


Use source distribution
-----------------------

BitcoinZ Electrum is a pure python application. If you want to use the
Qt interface, install the Qt dependencies::

    sudo apt-get install python3-pyqt5

If you downloaded the official package (tar.gz), you can run
BitcoinZ Electrum from its root directory without installing it on your
system; all the python dependencies are included in the 'packages'
directory (except x11-hash).

To install x11-hash dependency in the 'packages' dir run once::

    python3 -m pip install -t packages x11-hash

To install precise tested versions of HW libs (trezor, ledeger, etc) run once::

    python3 -m pip install -t packages -r contrib/deterministic-build/requirements-hw.txt

To install precise tested version of pyqt5 run once::

    python3 -m pip install -t packages -r contrib/deterministic-build/requirements-binaries.txt

To run BitcoinZ Electrum from its root directory, just do::

    ./electrum-btcz

You can also install BitcoinZ Electrum on your system, by running this command::

    sudo apt-get install python3-setuptools
    python3 -m pip install .[fast]

This will download and install the Python dependencies used by
BitcoinZ Electrum instead of using the 'packages' directory.
The 'fast' extra contains some optional dependencies that we think
are often useful but they are not strictly needed.

If you cloned the git repository, you need to compile extra files
before you can run BitcoinZ Electrum. Read the next section, "Development
Version".


Using Tor proxy
===============

Starting from BitcoinZ Electrum release 3.2.3.1 automatic Tor Proxy
detection and use on wallet startup is added to
`Network <docs/tor/tor-proxy-on-startup.md>`_ preferences.

To use Tor Proxy on Ubuntu set it up with::

    sudo apt-get install tor
    sudo service tor start

Other platforms setup is described at `docs/tor.md <docs/tor.md>`_

Development version
===================

Check out the code from GitHub::

    git clone https://github.com/btcz-electrum/electrum-btcz.git
    cd electrum-btcz

Run install (this should install dependencies)::

    python3 -m pip install .[fast]


Compile the protobuf description file::

    sudo apt-get install protobuf-compiler
    protoc --proto_path=electrum_btcz --python_out=electrum_btcz electrum_btcz/paymentrequest.proto

Create translations (optional)::

    sudo apt-get install python-requests gettext
    ./contrib/make_locale
