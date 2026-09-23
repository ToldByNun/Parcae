; Parcae Windows installer (Inno Setup 6).
; Build via packaging/windows/build_installer.sh (copies bootstrap into stage).
;
; Defines: ParcaeVersion, ParcaeFlavor (cpu|cuda|full), ParcaeStage, ParcaeOut

#ifndef ParcaeVersion
  #define ParcaeVersion "0.7.0"
#endif
#ifndef ParcaeFlavor
  #define ParcaeFlavor "cpu"
#endif
#ifndef ParcaeStage
  #error "ParcaeStage must be defined (staged payload directory)"
#endif
#ifndef ParcaeOut
  #define ParcaeOut "."
#endif

#if StrComp(ParcaeFlavor, "cpu", False) == 0
  #define ParcaeSuffix "-cpu"
  #define ParcaeAppName "Parcae (CPU)"
  #define ParcaeAppIdGuid "{{A7C3E5F1-9B2D-4E8A-B1C0-111111111111}"
#elif StrComp(ParcaeFlavor, "cuda", False) == 0
  #define ParcaeSuffix "-cuda"
  #define ParcaeAppName "Parcae (CUDA)"
  #define ParcaeAppIdGuid "{{A7C3E5F1-9B2D-4E8A-B1C0-222222222222}"
#else
  #define ParcaeSuffix ""
  #define ParcaeAppName "Parcae"
  #define ParcaeAppIdGuid "{{A7C3E5F1-9B2D-4E8A-B1C0-333333333333}"
#endif

#define MyAppPublisher "Parcae"
#define MyAppURL "https://github.com/ToldByNun/Parcae"
#define MyAppExeName "parcae-catalog.exe"

[Setup]
AppId={#ParcaeAppIdGuid}AppName={#ParcaeAppName}
AppVersion={#ParcaeVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\Parcae
DefaultGroupName=Parcae
DisableProgramGroupPage=no
LicenseFile={#ParcaeStage}\LICENSE
OutputDir={#ParcaeOut}
OutputBaseFilename=Parcae-v{#ParcaeVersion}-windows-x64{#ParcaeSuffix}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addpath"; Description: "Add Parcae bin directory to system PATH"; GroupDescription: "Environment"; Flags: checkedonce
Name: "pyeditable"; Description: "Install Python packages editable (python/ + agents/)"; GroupDescription: "Developer"; Flags: unchecked

[Files]
Source: "{#ParcaeStage}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#ParcaeAppName} Catalog"; Filename: "{app}\bin\{#MyAppExeName}"; Parameters: "--json"; WorkingDir: "{app}"
Name: "{group}\{#ParcaeAppName} Search Cycle (status)"; Filename: "{app}\bin\parcae-search-cycle.exe"; Parameters: "--status --json --data-dir ""{app}\data"""; WorkingDir: "{app}"
Name: "{group}\Developer shell"; Filename: "{cmd}"; Parameters: "/K set PATH={app}\bin;%PATH% && cd /d {app}\src"
Name: "{group}\README"; Filename: "{app}\README.md"
Name: "{group}\{cm:UninstallProgram,{#ParcaeAppName}}"; Filename: "{uninstallexe}"
#if StrComp(ParcaeFlavor, "full", False) == 0
Name: "{group}\Parcae (CUDA) Catalog"; Filename: "{app}\bin-cuda\{#MyAppExeName}"; Parameters: "--json"; WorkingDir: "{app}"
#endif
Name: "{autodesktop}\{#ParcaeAppName}"; Filename: "{app}\bin\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\packaging\windows\bootstrap.ps1"" -Flavor {#ParcaeFlavor}"; \
  StatusMsg: "Checking and installing build prerequisites (MSVC, CMake, Python, optional CUDA)..."; \
  Flags: runhidden waituntilterminated
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\install-python-editable.ps1"""; \
  StatusMsg: "Installing Python packages (editable)..."; \
  Flags: runhidden waituntilterminated; Tasks: pyeditable
Filename: "{app}\bin\{#MyAppExeName}"; Parameters: "--json"; Description: "Run parcae-catalog --json"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
const
  EnvironmentKey = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';

function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

procedure EnvAddPath(Path: string);
var
  OrigPath: string;
begin
  if not NeedsAddPath(Path) then
    exit;
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', OrigPath) then
    OrigPath := '';
  if OrigPath <> '' then
    Path := OrigPath + ';' + Path;
  RegWriteExpandStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', Path);
end;

procedure EnvRemovePath(Path: string);
var
  OrigPath, P: string;
  PosStart: Integer;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', OrigPath) then
    exit;
  P := ';' + OrigPath + ';';
  PosStart := Pos(';' + Path + ';', P);
  if PosStart = 0 then
    exit;
  Delete(P, PosStart + 1, Length(Path) + 1);
  if (Length(P) > 0) and (P[1] = ';') then
    Delete(P, 1, 1);
  if (Length(P) > 0) and (P[Length(P)] = ';') then
    Delete(P, Length(P), 1);
  RegWriteExpandStringValue(HKEY_LOCAL_MACHINE, EnvironmentKey, 'Path', P);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('addpath') then
      EnvAddPath(ExpandConstant('{app}\bin'));
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    EnvRemovePath(ExpandConstant('{app}\bin'));
end;
