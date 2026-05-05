#!/bin/bash
# A shell script for running a java class

ART=$ARTISYNTH_HOME
if [[ -z $ART ]]; then
    ART=".."
fi

# OS Detection
# Classpath needs to be set to main classes directory and jar files
# in $ARTISYNTH_HOME/lib
OSNAME=`uname -s`
if [ "$OSNAME" = "Linux" ] ; then
    if uname -m | grep -q 64 ; then
       OS=$OSNAME"64"
    else
       OS=$OSNAME
       MEM_LIMIT="-Xmx2G"
    fi
    contains $LD_LIBRARY_PATH $ARTISYNTH_HOME/lib/$OS
    if [ $? -ne 0 ] || [ -z $LD_LIBRARY_PATH ] ; then
        export LD_LIBRARY_PATH=$ART/lib/$OS:$LD_LIBRARY_PATH
    fi
    # On WSL2, Mesa's libGLX_mesa.so crashes with SIGSEGV inside
    # glXQueryDrawable when JOGL probes for swap-interval support.
    # Preload a shim that replaces glXQueryDrawable with a safe no-op,
    # intercepting both direct calls and the glXGetProcAddressARB path
    # that JOGL uses to resolve GLX extension function pointers.
    if grep -qi microsoft /proc/version 2>/dev/null; then
        WSL_FIX="$ART/lib/Linux64/libwsl2_glx_fix.so"
        WSL_SRC="$ART/support/linux/wsl2_glx_fix.c"
        if [ ! -f "$WSL_FIX" ] && [ -f "$WSL_SRC" ] && command -v gcc >/dev/null 2>&1; then
            gcc -shared -fPIC -O2 -o "$WSL_FIX" "$WSL_SRC" -ldl 2>/dev/null
        fi
        if [ -f "$WSL_FIX" ]; then
            export LD_PRELOAD="$WSL_FIX${LD_PRELOAD:+:$LD_PRELOAD}"
        fi
    fi
    PATH="$PATH:$ART/bin"
    CLASSPATH="$ART/classes:$ART/lib/*"
elif [ "$OSNAME" = "Darwin" ] ; then
    if [ `uname -p` = "powerpc" ] ; then
        ARCH="ppc"
    elif java -version 2>&1 | fgrep -q 'version "1.6' ; then
        ARCH="x86_64"
    else
        ARCH="i386"
    fi
    contains $DYLD_LIBRARY_PATH $ART/lib/$OSNAME-$ARCH
    if [ $? -ne 0 ] || [ -z $DYLD_LIBRARY_PATH ]; then
        export DYLD_LIBRARY_PATH=$ART/lib/$OSNAME-$ARCH:$DYLD_LIBRARY_PATH
    fi
    PATH="$PATH:$ART/bin"
    CLASSPATH="$ART/classes:$ART/lib/*"
elif echo "$OSNAME" | grep CYGWIN 1>/dev/null 2>&1 ; then
    export ARTISYNTH_HOME=`cygpath -w $ART`
    export ARTISYNTH_PATH=".;`cygpath -w $HOME`;$ART"
    if uname -m | grep -q 64 ; then
       # echo Cygwin on Windows64
       export PATH=`cygpath "$ART\lib\Windows64"`:$PATH
    else
       # echo Cygwin on Windows32
       export PATH=`cygpath "$ART\lib\Windows"`:$PATH
       MEM_LIMIT="-Xmx1G"
    fi
    PATH="$PATH:$ART/bin:"
    CLASSPATH="$ART\classes;$ART\lib\*"
else     
    echo Unknown operating system: $OSNAME
fi
export CLASSPATH
export PATH

java "$@" 
