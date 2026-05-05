/*
 * WSL2 GLX fix — intercepts the dlsym call chain used by JOGL/gluegen to
 * resolve glXQueryDrawable, replacing it with a safe no-op stub.
 *
 * Root cause: on WSL2 (WSLg XWayland), Mesa's libGLX_mesa.so crashes with
 * SIGSEGV (PC=0) inside glXQueryDrawable because the XWayland-backed drawable
 * has a null function pointer in its attribute-query vtable.  The crash
 * happens in JOGL's SharedResourceRunner before any ArtiSynth code runs.
 *
 * Resolution path that must be intercepted:
 *   libgluegen_rt.so:UnixDynamicLinkerImpl.dlsym(libGL, "glXGetProcAddressARB")
 *     → libjogl_desktop.so dispatch calls returned pointer("glXQueryDrawable")
 *     → Mesa glXQueryDrawable → SIGSEGV
 *
 * Fix: wrap dlsym so that any lookup for "glXGetProcAddressARB" returns our
 * wrapper instead of Mesa's.  Our wrapper transparently forwards all names
 * except "glXQueryDrawable", which it replaces with a safe stub.
 *
 * Build (no GL/X dev headers required):
 *   gcc -shared -fPIC -O2 -o libwsl2_glx_fix.so wsl2_glx_fix.c -ldl
 *
 * Usage (set automatically by runapp.sh on WSL2):
 *   LD_PRELOAD=/path/to/libwsl2_glx_fix.so
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

typedef void (*fp_t)(void);
typedef fp_t (*gpa_fn_t)(const unsigned char *);

/* Safe stub: return 0 instead of crashing inside Mesa */
static void wsl2_safe_glXQueryDrawable(
    void *dpy, unsigned long draw, int attribute, unsigned int *value)
{
    (void)dpy; (void)draw; (void)attribute;
    if (value)
        *value = 0;
}

/*
 * Wrapper for glXGetProcAddressARB/glXGetProcAddress.
 * Replaces glXQueryDrawable with the safe stub; forwards everything else
 * to the real Mesa implementation saved at first interception.
 */
static gpa_fn_t real_gpa = NULL;

static fp_t wsl2_glXGetProcAddressARB(const unsigned char *procName)
{
    if (procName && strcmp((const char *)procName, "glXQueryDrawable") == 0)
        return (fp_t)wsl2_safe_glXQueryDrawable;
    return real_gpa ? real_gpa(procName) : NULL;
}

/*
 * Wrap dlsym so libgluegen_rt.so gets our glXGetProcAddressARB wrapper
 * when it resolves it from libGL.
 *
 * Bootstrap: use dlvsym(RTLD_NEXT, ...) to obtain the real dlsym without
 * triggering recursion into our own wrapper.
 */
void *dlsym(void *handle, const char *symbol)
{
    typedef void *(*dlsym_fn_t)(void *, const char *);
    static dlsym_fn_t real_dlsym = NULL;
    if (!real_dlsym)
        real_dlsym = (dlsym_fn_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym)
        return NULL;

    if (symbol) {
        /* Intercept direct glXQueryDrawable lookups too */
        if (strcmp(symbol, "glXQueryDrawable") == 0)
            return (void *)wsl2_safe_glXQueryDrawable;

        /* Return our glXGetProcAddressARB wrapper; save the real pointer */
        if (strcmp(symbol, "glXGetProcAddressARB") == 0 ||
            strcmp(symbol, "glXGetProcAddress") == 0) {
            void *real = real_dlsym(handle, symbol);
            if (real && !real_gpa)
                real_gpa = (gpa_fn_t)real;
            return (void *)wsl2_glXGetProcAddressARB;
        }
    }

    return real_dlsym(handle, symbol);
}
