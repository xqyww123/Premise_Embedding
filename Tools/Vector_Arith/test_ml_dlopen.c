/* Guards the invariant that isabelle_vector.cpp documents: Isabelle/ML must be able
 * to load this shared object into a process that has no Python in it.
 *
 * The kernel no longer references CPython at all -- the glue moved to
 * Isabelle_Semantic_Embedding/_vecgather.c, a separate extension module -- so this
 * now guards against a regression rather than against the design. -Wl,-z,defs is the
 * build-time half of the same guard; this is the load-time half, and it is the only
 * one that would catch a Python symbol arriving via a static library.
 *
 * This reproduces exactly what Poly/ML does, then calls the entry points ML uses,
 * so such a mistake fails the build instead of Isabelle:
 *
 *   Unix    dlopen(path, RTLD_LAZY)   libpolyml/polyffi.cpp:167
 *   Windows LoadLibrary(path)         libpolyml/polyffi.cpp:153
 *
 * Lazy binding defers *function* symbols, but data symbols relocate eagerly even so:
 * a single Py_None or PyList_Type reference would break the ML side outright at load.
 * LoadLibrary is stricter still, resolving the whole ordinary import table -- which
 * is why a python3.dll import there depends on Python happening to sit on the
 * process's DLL search path, and the ML process is launched from a Cygwin shell
 * whose PATH we do not control. Passing here on Windows shows the kernel needs no
 * such luck; it cannot show that Poly/ML itself is happy, since no Isabelle is
 * installed on the CI runner. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#define DEFAULT_LIB "./isabelle_vector.dll"
#else
#include <dlfcn.h>
#define DEFAULT_LIB "./libisabelle_vector.so"
#endif

typedef int16_t (*dot_fn)(const int16_t *, const int16_t *, size_t);

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : DEFAULT_LIB;
  dot_fn dot;
  void *topk;

#ifdef _WIN32
  HMODULE h = LoadLibraryA(path);
  if (!h) {
    fprintf(stderr, "FAIL: Isabelle/ML could not LoadLibrary %s\n  error %lu\n", path,
            (unsigned long)GetLastError());
    fprintf(stderr, "  Error 126 (ERROR_MOD_NOT_FOUND) means a dependency in the import\n"
                    "  table could not be resolved -- python3.dll, or a MinGW runtime.\n");
    return 1;
  }
  /* GetProcAddress returns FARPROC; the cast through void* is the documented way to
     reach a data-shaped function pointer without tripping -Wcast-function-type. */
  dot = (dot_fn)(void *)GetProcAddress(h, "dot_q15");
  topk = (void *)GetProcAddress(h, "top_k_q15");
#else
  void *h = dlopen(path, RTLD_LAZY);
  if (!h) {
    fprintf(stderr, "FAIL: Isabelle/ML could not dlopen %s\n  %s\n", path, dlerror());
    fprintf(stderr, "  A Python *data* symbol (Py_None, PyList_Type, PyExc_*) most\n"
                    "  likely crept in; only Python *functions* may be referenced.\n");
    return 1;
  }
  dot = (dot_fn)dlsym(h, "dot_q15");
  topk = dlsym(h, "top_k_q15");
#endif

  if (!dot || !topk) {
    fprintf(stderr, "FAIL: dot_q15 / top_k_q15 not resolvable\n");
    fprintf(stderr, "  On Windows, check the symbols are exported undecorated:\n"
                    "  extern \"C\" on x64 adds no decoration, but a missing\n"
                    "  WINDOWS_EXPORT_ALL_SYMBOLS would export nothing at all.\n");
    return 1;
  }

  int16_t a[64], b[64];
  for (int i = 0; i < 64; i++) { a[i] = (int16_t)(i * 37); b[i] = (int16_t)(i * 11); }
  volatile int16_t s = dot(a, b, 64); /* actually bind and call, as ML would */
  (void)s;

#ifdef _WIN32
  printf("PASS: LoadLibrary + dot_q15 work without Python in the process\n");
  FreeLibrary(h);
#else
  printf("PASS: dlopen(RTLD_LAZY) + dot_q15 work without Python in the process\n");
  dlclose(h);
#endif
  return 0;
}
