; Windows setup for BTCZ Electrum, built on GitHub Actions (.github/workflows/build.yml):
;   makensis /NOCD /DPRODUCT_VERSION=<version> contrib\github\btcz-electrum-setup.nsi
; run from the repository root after "pyinstaller win.spec" with BTCZ_WIN_ONEDIR=1.
;
; Installing over any earlier BTCZ Electrum (this setup, or the old 3.x installer) first removes
; the old program files and shortcuts. Wallets in %APPDATA%\Electrum-BTCZ (and the old
; %APPDATA%\Electrum-ZCASH) are never touched, neither by the setup nor by the uninstaller.

  Unicode true
  !include "MUI2.nsh"
  !include "x64.nsh"
  !include "LogicLib.nsh"
  !include "FileFunc.nsh"
  !include "StrFunc.nsh"
  ${StrStr}

  !define PRODUCT_NAME "BitcoinZ Electrum"
  !define PRODUCT_EXE "BTCZ-Electrum.exe"
  !define PRODUCT_WEB_SITE "https://github.com/RokMikuz/btcz-electrum-wallet"
  !define PRODUCT_PUBLISHER "BitcoinZ Community"
  !define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
  ; names used by older BTCZ Electrum installers
  !define PREV_PROD_NAME "Electrum-ZCASH"
  !define PREV_PROD_NAME2 "BTCZ-Electrum"

  Var PREVINSTDIR

  Name "${PRODUCT_NAME}"
  OutFile "dist\BTCZ-Electrum-${PRODUCT_VERSION}-setup.exe"
  InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"
  RequestExecutionLevel admin
  SetCompressor /SOLID lzma
  ShowInstDetails show
  ShowUninstDetails show
  BrandingText "${PRODUCT_NAME} ${PRODUCT_VERSION}"
  Caption "${PRODUCT_NAME} ${PRODUCT_VERSION} Setup"

  VIProductVersion 1.0.0.0
  VIAddVersionKey ProductName "${PRODUCT_NAME}"
  VIAddVersionKey CompanyName "${PRODUCT_PUBLISHER}"
  VIAddVersionKey FileDescription "${PRODUCT_NAME} Setup"
  VIAddVersionKey FileVersion "${PRODUCT_VERSION}"
  VIAddVersionKey ProductVersion "${PRODUCT_VERSION}"
  VIAddVersionKey LegalCopyright "MIT License"

  !define MUI_ICON "electrum_btcz\gui\icons\electrum-btcz.ico"
  !define MUI_UNICON "electrum_btcz\gui\icons\electrum-btcz.ico"
  !define MUI_ABORTWARNING

  !define MUI_WELCOMEPAGE_TEXT "This will install ${PRODUCT_NAME} ${PRODUCT_VERSION}.$\r$\n$\r$\nAn older ${PRODUCT_NAME} installed on this computer is removed first. Your wallets are not affected: they stay in %APPDATA%\Electrum-BTCZ.$\r$\n$\r$\nPlease close ${PRODUCT_NAME} before continuing."
  !insertmacro MUI_PAGE_WELCOME
  !insertmacro MUI_PAGE_DIRECTORY
  !insertmacro MUI_PAGE_INSTFILES
  !define MUI_FINISHPAGE_TEXT "${PRODUCT_NAME} ${PRODUCT_VERSION} is installed. Start it from the desktop or the Start menu."
  !insertmacro MUI_PAGE_FINISH

  !define MUI_UNCONFIRMPAGE_TEXT_TOP "${PRODUCT_NAME} will be removed. Your wallet files in %APPDATA%\Electrum-BTCZ are kept."
  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_INSTFILES

  !insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "${PRODUCT_NAME} needs 64-bit Windows."
    Abort
  ${EndIf}
  SetRegView 64
  ; install where the previous version was, if there is one
  ReadRegStr $0 HKLM "${PRODUCT_UNINST_KEY}" "InstallLocation"
  ${If} $0 != ""
    StrCpy $INSTDIR $0
  ${EndIf}
FunctionEnd

Function un.onInit
  SetRegView 64
FunctionEnd

; Removes an older installation folder, but only if it really is one (has the old uninstaller
; or our exe in it), so a wrong registry value can never delete an unrelated folder.
!macro RemoveOldInstall DIR
  ${If} ${DIR} != ""
  ${AndIf} ${FileExists} "${DIR}\*.*"
    ${If} ${FileExists} "${DIR}\Uninstall.exe"
    ${OrIf} ${FileExists} "${DIR}\${PRODUCT_EXE}"
      DetailPrint "Removing old version in ${DIR}"
      RMDir /r "${DIR}"
    ${EndIf}
  ${EndIf}
!macroend

!macro RemoveShortcuts NAME
  SetShellVarContext current
  Delete "$DESKTOP\${NAME}.lnk"
  RMDir /r "$SMPROGRAMS\${NAME}"
  SetShellVarContext all
  Delete "$DESKTOP\${NAME}.lnk"
  RMDir /r "$SMPROGRAMS\${NAME}"
!macroend

Section "${PRODUCT_NAME}" SecMain
  ; always install into a folder of our own, so removing it can never take other files along
  StrLen $1 "${PRODUCT_NAME}"
  IntOp $1 0 - $1
  StrCpy $0 $INSTDIR "" $1
  ${If} $0 != "${PRODUCT_NAME}"
    StrCpy $INSTDIR "$INSTDIR\${PRODUCT_NAME}"
  ${EndIf}

  ; make sure the wallet is not running, its files can not be replaced otherwise
  check_running:
  ${If} ${FileExists} "$INSTDIR\${PRODUCT_EXE}"
    ClearErrors
    FileOpen $0 "$INSTDIR\${PRODUCT_EXE}" a
    ${If} ${Errors}
      MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "${PRODUCT_NAME} is still running. Please close it and click Retry." IDRETRY check_running
      Abort
    ${EndIf}
    FileClose $0
  ${EndIf}

  ; older installers (3.x) kept the install folder in HKCU\Software\<name>
  ReadRegStr $PREVINSTDIR HKCU "Software\${PREV_PROD_NAME}" ""
  !insertmacro RemoveOldInstall $PREVINSTDIR
  ReadRegStr $PREVINSTDIR HKCU "Software\${PREV_PROD_NAME2}" ""
  !insertmacro RemoveOldInstall $PREVINSTDIR
  ReadRegStr $PREVINSTDIR HKCU "Software\${PRODUCT_NAME}" ""
  !insertmacro RemoveOldInstall $PREVINSTDIR
  ReadRegStr $PREVINSTDIR HKLM "${PRODUCT_UNINST_KEY}" "InstallLocation"
  !insertmacro RemoveOldInstall $PREVINSTDIR
  !insertmacro RemoveOldInstall $INSTDIR

  !insertmacro RemoveShortcuts "${PREV_PROD_NAME}"
  !insertmacro RemoveShortcuts "${PREV_PROD_NAME2}"
  !insertmacro RemoveShortcuts "${PRODUCT_NAME}"
  DeleteRegKey HKCU "Software\${PREV_PROD_NAME}"
  DeleteRegKey HKCU "Software\${PREV_PROD_NAME2}"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PREV_PROD_NAME}"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PREV_PROD_NAME2}"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
  ; the 3.x installer linked zcash: URIs to the BTCZ wallet; drop that link only if it is ours
  ReadRegStr $0 HKCU "Software\Classes\zcash\shell\open\command" ""
  ${StrStr} $1 $0 "electrum-btcz"
  ${If} $1 != ""
    DeleteRegKey HKCU "Software\Classes\zcash"
  ${EndIf}

  SetOutPath "$INSTDIR"
  File /r "dist\BTCZ-Electrum\*.*"
  File "electrum_btcz\gui\icons\electrum-btcz.ico"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  SetShellVarContext current
  CreateShortCut "$DESKTOP\${PRODUCT_NAME}.lnk" "$INSTDIR\${PRODUCT_EXE}" "" "$INSTDIR\${PRODUCT_EXE}" 0
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\${PRODUCT_EXE}" "" "$INSTDIR\${PRODUCT_EXE}" 0
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0

  ; bitcoinz: payment links open the wallet
  WriteRegStr HKCU "Software\Classes\bitcoinz" "" "URL:bitcoinz Protocol"
  WriteRegStr HKCU "Software\Classes\bitcoinz" "URL Protocol" ""
  WriteRegStr HKCU "Software\Classes\bitcoinz\DefaultIcon" "" "$\"$INSTDIR\electrum-btcz.ico$\",0"
  WriteRegStr HKCU "Software\Classes\bitcoinz\shell\open\command" "" "$\"$INSTDIR\${PRODUCT_EXE}$\" $\"%1$\""

  WriteRegStr HKCU "Software\${PRODUCT_NAME}" "" "$INSTDIR"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\electrum-btcz.ico"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
  WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

Section "Uninstall"
  ; program files only; %APPDATA%\Electrum-BTCZ (the wallets) stays
  ${If} ${FileExists} "$INSTDIR\${PRODUCT_EXE}"
    RMDir /r "$INSTDIR"
  ${Else}
    Delete "$INSTDIR\Uninstall.exe"
    RMDir "$INSTDIR"
  ${EndIf}
  !insertmacro RemoveShortcuts "${PRODUCT_NAME}"
  DeleteRegKey HKCU "Software\Classes\bitcoinz"
  DeleteRegKey HKCU "Software\${PRODUCT_NAME}"
  DeleteRegKey HKLM "${PRODUCT_UNINST_KEY}"
SectionEnd
