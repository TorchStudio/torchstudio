# 2>nul&@goto :BATCH

#BASH

SCRIPTDIR=$(cd "$(dirname "$0")"; pwd)
(
cd "$SCRIPTDIR"
cd ..

pythonpath="$(pwd)/python"
channel="pytorch"
cuda=""
packages=""
uninstall=""
while [ "$1" != "" ]; do
    if [ $1 == "--path" ]; then
        shift; pythonpath=$1
    elif [ $1 == "--channel" ]; then
        shift; channel=$1
    elif [ $1 == "--cuda" ]; then
        cuda="--cuda"
    elif [ $1 == "--package" ]; then
        shift; packages+="--package $1"
    elif [ $1 == "--uninstall" ]; then
        uninstall="1"
    fi
    shift
done

if [ ! -z "$uninstall" ]; then
    echo "Uninstalling python environment..."
    rm -f *.sh
    rm -f *.tmp
    rm -f -r "$pythonpath"
    if [ $? != 0 ]; then
        echo "" 1>&2
        echo "Error while uninstalling. Check write permissions." 1>&2
        exit 1
    else
        echo ""
        echo "Uninstall complete."
        exit 0
    fi
fi

if [[ $OSTYPE == "linux"* ]]; then
    echo "Downloading, installing and setting up a linux python environment"
elif [[ $OSTYPE == "darwin"* ]]; then
    echo "Downloading, installing and setting up a macOS python environment"
else
    echo "Error: unsupported OS ($OSTYPE)" 1>&2
    exit 1
fi

if [ ! -z "$cuda" ]; then
    echo "This may take up to 16 minutes depending on your download speed, and up to 16 GB."
else
    echo "This may take up to 5 minutes depending on your download speed, and up to 5 GB."
fi

echo ""
if [[ $OSTYPE == "linux"* ]]; then
    file=Miniconda3-latest-Linux-x86_64.sh
    rm -f "$file.tmp"
    if [ -f "$file" ]; then
        echo "Python installer ($file) already downloaded"
    else
        echo "Downloading Python installer ($file)..."
        wget --show-progress --progress=bar:force:noscroll --no-check-certificate https://repo.anaconda.com/miniconda/$file -O "$file.tmp"
        if [ $? != 0 ]; then
            rm -f "$file.tmp"
            echo "" 1>&2
            echo "Error while downloading. Make sure port 80 is open." 1>&2
            exit 1
        else
            mv "$file.tmp" "$file"
        fi
    fi
elif [[ $OSTYPE == "darwin"* ]]; then
    if [ "$(uname -m)" == "arm64" ]; then
        file=Miniconda3-latest-MacOSX-arm64.sh
    else
        file=Miniconda3-latest-MacOSX-x86_64.sh
    fi
    rm -f "$file.tmp"
    if [ -f "$file" ]; then
        echo "Python installer $file already downloaded"
    else
        echo "Downloading Python installer $file..."
        curl --insecure https://repo.anaconda.com/miniconda/$file -o "$file.tmp"
        if [ $? != 0 ]; then
            rm -f "$file.tmp"
            echo "" 1>&2
            echo "Error while downloading. Make sure port 80 is open." 1>&2
            exit 1
        else
            mv "$file.tmp" "$file"
        fi
    fi
fi

echo ""
if [ -f "python.tmp" ]; then
    rm -f python.tmp
    rm -f -r "$pythonpath"
fi
if [ -d "$pythonpath" ]; then
    echo "Python already installed in $pythonpath"
else
    echo "Installing Python in $pythonpath..."
    echo "" > python.tmp
    bash "$(pwd)/$file" -b -f -p "$pythonpath"
    rm -f python.tmp
    if [ $? != 0 ]; then
        rm -f -r "$pythonpath"
        echo "" 1>&2
        echo "Error while installing. Make sure you have write permissions." 1>&2
        exit 1
    fi
fi

PATH="$PATH;$pythonpath/bin"
"$pythonpath/bin/python" -u -B -X utf8 -m torchstudio.pythoninstall --channel $channel $cuda $packages
if [ $? != 0 ]; then
    echo "" 1>&2
    echo "Error while installing packages" 1>&2
    exit 1
fi

echo ""
echo "Installation complete."
)
exit


:BATCH

@echo off
setlocal
cd /D "%~dp0"
cd ..

set pythonpath=%cd%\python
set channel=pytorch
set cuda=
set packages=
set uninstall=
:args
if "%~1" == "--path" (
    set pythonpath=%~2
    shift
) else if "%~1" == "--channel" (
    set channel=%~2
    shift
) else if "%~1" == "--cuda" (
    set cuda=--cuda
) else if "%~1" == "--package" (
    set packages=%packages% --package %~2
    shift
) else if "%~1" == "--uninstall" (
    set uninstall=1
) else if "%~1" == "" (
    goto endargs
)
shift
goto args
:endargs

if DEFINED uninstall (
    echo Uninstalling python environment...
    del *.exe 2>nul
    del *.tmp 2>nul
    rmdir /s /q "%pythonpath%" 2>nul
    if ERRORLEVEL 1 (
        echo. 1>&2
        echo Error while uninstalling. Check write permissions. 1>&2
        exit /B 1
    ) else (
        echo.
        echo Uninstall complete.
        exit /B 0
    )
)

echo Downloading, installing and setting up a windows python environment
if DEFINED cuda (
    echo This may take up to 16 minutes depending on your download speed, and up to 16 GB.
) else (
    echo This may take up to 5 minutes depending on your download speed, and up to 5 GB.
)

echo.
set file=Miniconda3-latest-Windows-x86_64.exe
del %file%.tmp 2>nul
if EXIST "%file%" (
    echo Python installer %file% already downloaded
) else (
    echo Downloading Python installer %file%...
    curl --insecure https://repo.anaconda.com/miniconda/%file% -o %file%.tmp
    if ERRORLEVEL 1 (
        del %file%.tmp 2>nul
        echo. 1>&2
        echo Error while downloading. Make sure port 80 is open. 1>&2
        exit /B 1
    ) else (
        ren %file%.tmp %file%
    )
)

echo.
if EXIST "python.tmp" (
    del python.tmp 2>nul
    rmdir /s /q "%pythonpath%" 2>nul
)
if EXIST "%pythonpath%" (
    echo Python already installed in %pythonpath%
) else (
    echo Installing Python in %pythonpath%...
    echo. > python.tmp
    %file% /S /D=%pythonpath%
    del python.tmp 2>nul
    if ERRORLEVEL 1 (
        rmdir /s /q "%pythonpath%" 2>nul
        echo. 1>&2
        echo Error while installing. Make sure you have write permissions. 1>&2
        exit /B 1
    )
)

set PATH=%PATH%;%pythonpath%;%pythonpath%\Library\mingw-w64\bin;%pythonpath%\Library\bin
"%pythonpath%\python" -u -B -X utf8 -m torchstudio.pythoninstall --channel %channel% %cuda% %packages%
if ERRORLEVEL 1 (
    echo. 1>&2
    echo Error while installing packages 1>&2
    exit /B 1
)

echo.
echo Installation complete.
