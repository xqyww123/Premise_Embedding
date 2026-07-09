/* Batch extraction of raw buffer addresses from a list of Python objects.
 *
 * topk must tell the SIMD kernel where each candidate vector lives inside the
 * LMDB mmap. A memoryview does not expose its pointer, so Python's only recourse
 * is to wrap each one -- np.frombuffer(mv, uint8).ctypes.data -- purely to read
 * the address back out. That costs ~2.9us per key; over a 10^5 domain it is
 * ~230ms of pure overhead, more than the SIMD scan it feeds. The same loop here
 * runs at ~0.1us per key.
 *
 * This is a CPython extension module, imported rather than dlopened. It used to
 * live inside libisabelle_vector.so, which Isabelle/ML also loads through
 * Foreign.loadLibrary -- into a process with no Python. That forced two rules on
 * the old file, and BOTH ARE NOW OBSOLETE:
 *
 *   - "reference Python functions but never Python data": Py_RETURN_NONE below
 *     touches _Py_NoneStruct, a data symbol. Perfectly fine here. The rule moved
 *     to the kernel (Tools/Vector_Arith/isabelle_vector.cpp), which now contains
 *     no Python at all -- an invariant of the *file*, not of the programmer.
 *   - "call through ctypes.PyDLL, never CDLL": there is no ctypes any more. An
 *     extension module's entry point holds the GIL by construction, which is what
 *     the C API calls below require. The old CDLL/PyDLL split -- and the SIGSEGV
 *     it once caused -- cannot recur.
 *
 * Holding the GIL costs nothing at the call site: gather_addrs runs ~10ms over a
 * 10^5 domain, while top_k_q15_gather -- which does release it, being a plain
 * ctypes.CDLL call into the kernel -- runs ~150ms. The event-loop stall that
 * asyncio.to_thread exists to avoid is set by the latter.
 *
 * Python 3.11 is the floor: Py_buffer and PyObject_GetBuffer entered the limited
 * API only there ("Limited API and stable ABI since Python 3.11", pybuffer.h).
 * The other three -- PyList_Size, PyList_GetItem, PyBuffer_Release, PyErr_Clear --
 * are available at that level too, so one abi3 build serves 3.11 through 3.14.
 *
 * The caller keeps the buffers alive and the LMDB read transaction open: the
 * addresses point into that transaction's MVCC snapshot of the mmap.
 */
#define Py_LIMITED_API 0x030B0000
#include <Python.h>
#include <stdint.h>

/* items    : list whose elements are buffer-exporting objects (memoryviews from a
 *            buffers=True transaction), or None where the key had no record.
 * expected : required byte length. Anything else is counted as skipped: a stale
 *            float32 record is D*4 bytes and would otherwise be read as a
 *            truncated vector, silently.
 * out_addrs[j], out_keep[j] : address, and index into `items`, of the j-th kept item.
 * out_missing[j]            : index into `items` of the j-th item without a buffer.
 * counts                    : {kept, missing, skipped}.  All three out arrays must
 *                             have room for len(items).
 * Returns 0, or -1 if `items` is not a list. */
static int gather_addrs(PyObject *items, Py_ssize_t expected,
                        uintptr_t *out_addrs, int32_t *out_keep, int32_t *out_missing,
                        int32_t *counts) {
  const Py_ssize_t n = PyList_Size(items);
  if (n < 0) { PyErr_Clear(); return -1; }
  int32_t kept = 0, missing = 0, skipped = 0;
  Py_buffer view;
  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject *o = PyList_GetItem(items, i); /* borrowed */
    if (o == NULL) { PyErr_Clear(); skipped++; continue; }
    if (PyObject_GetBuffer(o, &view, PyBUF_SIMPLE) != 0) {
      PyErr_Clear();                 /* None, or anything else without a buffer */
      out_missing[missing++] = (int32_t)i;
      continue;
    }
    if (view.len != expected) { PyBuffer_Release(&view); skipped++; continue; }
    out_addrs[kept] = (uintptr_t)view.buf;
    out_keep[kept] = (int32_t)i;
    kept++;
    /* Releasing the view leaves the address valid: the memoryview still holds the
       mmap, which outlives the read transaction the caller keeps open. */
    PyBuffer_Release(&view);
  }
  counts[0] = kept; counts[1] = missing; counts[2] = skipped;
  return 0;
}

/* The out-parameters arrive as integers -- numpy's arr.ctypes.data -- because the
 * Python wrapper already allocates those arrays and already knows their addresses.
 * Taking them through the buffer protocol instead would be more idiomatic and
 * would rewrite a wrapper that is correct today; the pointers cross into C either
 * way. "K" is unsigned long long, wide enough for a pointer on every platform we
 * target; the double cast silences the -Wint-to-pointer-cast that a 32-bit build
 * would otherwise deserve. */
static PyObject *py_gather_addrs(PyObject *self, PyObject *args) {
  PyObject *items;
  Py_ssize_t expected;
  unsigned long long addrs, keep, missing, counts;
  (void)self;
  if (!PyArg_ParseTuple(args, "OnKKKK", &items, &expected,
                        &addrs, &keep, &missing, &counts))
    return NULL;
  if (gather_addrs(items, expected, (uintptr_t *)(uintptr_t)addrs,
                   (int32_t *)(uintptr_t)keep, (int32_t *)(uintptr_t)missing,
                   (int32_t *)(uintptr_t)counts) != 0) {
    PyErr_SetString(PyExc_TypeError, "gather_addrs expects a list");
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"gather_addrs", py_gather_addrs, METH_VARARGS,
     "gather_addrs(buffers, expected, addrs, keep, missing, counts) -> None"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_vecgather", NULL, -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__vecgather(void) { return PyModule_Create(&moduledef); }
