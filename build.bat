@echo off
setlocal enabledelayedexpansion
set "ROOT=%~dp0"

rem Locate vcvars64.bat. Deliberately avoids vswhere-through-a-for-loop: the
rem parenthesis in "Program Files (x86)" terminates the for clause early.
rem Delayed expansion sidesteps that, since the value is substituted post-parse.
set "PFX64=%ProgramFiles%"
set "PFX86=%ProgramFiles(x86)%"
set "VCVARS="
for %%E in (2026 2025 2022) do (
    for %%D in (Community Professional Enterprise BuildTools Preview) do (
        if not defined VCVARS (
            if exist "!PFX64!\Microsoft Visual Studio\%%E\%%D\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS=!PFX64!\Microsoft Visual Studio\%%E\%%D\VC\Auxiliary\Build\vcvars64.bat"
            )
        )
        if not defined VCVARS (
            if exist "!PFX86!\Microsoft Visual Studio\%%E\%%D\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS=!PFX86!\Microsoft Visual Studio\%%E\%%D\VC\Auxiliary\Build\vcvars64.bat"
            )
        )
    )
)

if not defined VCVARS (
    echo ERROR: no Visual Studio C++ x64 toolset found.
    echo Looked under "!PFX64!\Microsoft Visual Studio" and "!PFX86!\Microsoft Visual Studio".
    exit /b 1
)

call "!VCVARS!" >nul
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed: !VCVARS!
    exit /b 1
)

if not exist "%ROOT%build" mkdir "%ROOT%build"
if not exist "%ROOT%build\app" mkdir "%ROOT%build\app"
if not exist "%ROOT%build\tests" mkdir "%ROOT%build\tests"
if not exist "%ROOT%out" mkdir "%ROOT%out"

set "CFG=%~1"
if "%CFG%"=="" set "CFG=release"

if /i "%CFG%"=="debug" (
    set "OPT=/Od /Zi /MDd /D_DEBUG"
    set "NGXLIB=%ROOT%third_party\nvngx\lib\nvsdk_ngx_d_dbg.lib"
) else (
    set "OPT=/O2 /MD /DNDEBUG"
    set "NGXLIB=%ROOT%third_party\nvngx\lib\nvsdk_ngx_d.lib"
)

if not exist "!NGXLIB!" (
    echo ERROR: missing !NGXLIB!
    exit /b 1
)

set "COMMON=/nologo /std:c++17 /EHsc /W3 !OPT! /D_CRT_SECURE_NO_WARNINGS"
set "INCLUDES=/I "%ROOT%third_party\nvngx\include" /I "%ROOT%third_party\stb" /I "%ROOT%third_party\nvof" /I "%ROOT%src" /I "%ROOT%forwarder""
set "LIBS=d3d12.lib dxgi.lib d3dcompiler.lib dxguid.lib advapi32.lib user32.lib version.lib shell32.lib"
set "SHARED=%ROOT%src\common.cpp %ROOT%src\image.cpp %ROOT%src\gpu.cpp %ROOT%src\dlss.cpp %ROOT%src\cli.cpp %ROOT%src\pipeline.cpp %ROOT%src\nr.cpp %ROOT%src\optflow.cpp %ROOT%src\optflow_nvof.cpp %ROOT%src\slprobe.cpp"

echo Building video2dlssnr [%CFG%]...
cl !COMMON! !INCLUDES! ^
   /Fo"%ROOT%build\app\\" ^
   /Fd"%ROOT%build\video2dlssnr.pdb" ^
   /Fe"%ROOT%out\video2dlssnr.exe" ^
   !SHARED! "%ROOT%src\main.cpp" ^
   /link "!NGXLIB!" !LIBS!

if errorlevel 1 (
    echo.
    echo BUILD FAILED ^(video2dlssnr^)
    exit /b 1
)

echo Building video2dlssnr_tests [%CFG%]...
cl !COMMON! !INCLUDES! ^
   /Fo"%ROOT%build\tests\\" ^
   /Fd"%ROOT%build\video2dlssnr_tests.pdb" ^
   /Fe"%ROOT%out\video2dlssnr_tests.exe" ^
   !SHARED! "%ROOT%tests\test_main.cpp" ^
   /link "!NGXLIB!" !LIBS!

if errorlevel 1 (
    echo.
    echo BUILD FAILED ^(tests^)
    exit /b 1
)

echo Building forwarder [nvngx.dll_dlssnr.dll]...
cl !COMMON! /LD ^
   /Fo"%ROOT%build\app\fwd_" ^
   /Fe"%ROOT%out\nvngx.dll_dlssnr.dll" ^
   "%ROOT%forwarder\nvngx_fwd.cpp" ^
   /link kernel32.lib

if errorlevel 1 (
    echo.
    echo BUILD FAILED ^(forwarder^)
    exit /b 1
)

echo.
echo OK -^> %ROOT%out\video2dlssnr.exe
echo OK -^> %ROOT%out\video2dlssnr_tests.exe
echo OK -^> %ROOT%out\nvngx.dll_dlssnr.dll
endlocal
