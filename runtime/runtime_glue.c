/*
 * Compatibility translation unit for existing build commands.
 * Runtime functions are implemented in runtime_v5.c; defining wrappers here
 * would produce duplicate symbols when both files are linked.
 */
#include "runtime_v5.h"
