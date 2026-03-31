@echo off
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set WINSDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set WINSDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0

echo Building ray_tracer.dll...

"%MSVC%\bin\Hostx64\x64\cl.exe" /LD /O2 /EHsc /MD ^
    /I"%MSVC%\include" ^
    /I"%WINSDK_INC%\ucrt" ^
    src\ray_tracer.c ^
    /link ^
    /LIBPATH:"%MSVC%\lib\x64" ^
    /LIBPATH:"%WINSDK_LIB%\ucrt\x64" ^
    /LIBPATH:"%WINSDK_LIB%\um\x64" ^
    /OUT:lib\ray_tracer.dll

if %ERRORLEVEL% EQU 0 (
    echo Build succeeded!
    dir lib\ray_tracer.dll
) else (
    echo Build FAILED
)
