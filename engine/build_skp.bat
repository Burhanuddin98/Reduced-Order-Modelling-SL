@echo off
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set WINSDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set WINSDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0
set SKP_SDK=C:\RoomGUI\ROM\engine\sketchup_sdk

echo Building skp_reader.dll...

"%MSVC%\bin\Hostx64\x64\cl.exe" /LD /O2 /EHsc /MD ^
    /I"include" ^
    /I"%SKP_SDK%\headers" ^
    /I"%MSVC%\include" ^
    /I"%WINSDK_INC%\ucrt" ^
    /I"%WINSDK_INC%\shared" ^
    /I"%WINSDK_INC%\um" ^
    src\skp_reader.cpp ^
    /link ^
    /LIBPATH:"%SKP_SDK%\binaries\sketchup\x64" ^
    /LIBPATH:"%MSVC%\lib\x64" ^
    /LIBPATH:"%WINSDK_LIB%\ucrt\x64" ^
    /LIBPATH:"%WINSDK_LIB%\um\x64" ^
    SketchUpAPI.lib ^
    /OUT:lib\skp_reader.dll

if %ERRORLEVEL% EQU 0 (
    echo Build succeeded!
    copy "%SKP_SDK%\binaries\sketchup\x64\SketchUpAPI.dll" lib\ >/dev/null
    copy "%SKP_SDK%\binaries\sketchup\x64\SketchUpCommonPreferences.dll" lib\ >/dev/null
    dir lib\skp_reader.dll
) else (
    echo Build FAILED
)
