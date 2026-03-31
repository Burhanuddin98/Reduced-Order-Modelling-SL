@echo off
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set WINSDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set WINSDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0

echo Building helmholtz.dll...

"%CUDA%\bin\nvcc.exe" -shared -o helmholtz.dll helmholtz_gpu.cu -arch=sm_75 -ccbin "%MSVC%\bin\Hostx64\x64" -I"%MSVC%\include" -I"%WINSDK_INC%\ucrt" -I"%CUDA%\include" -L"%MSVC%\lib\x64" -L"%WINSDK_LIB%\ucrt\x64" -L"%WINSDK_LIB%\um\x64" -L"%CUDA%\lib\x64" -lcusolver -lcusparse -lcudart --compiler-options "/MD /EHsc" -DHELMHOLTZ_EXPORTS

if %ERRORLEVEL% EQU 0 (
    echo Build succeeded!
) else (
    echo Build FAILED
)
