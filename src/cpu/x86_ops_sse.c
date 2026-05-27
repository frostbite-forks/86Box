/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          SSE state and initialisation.
 *
 * Authors: frostbite3000
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
#include <stdint.h>
#include <string.h>
#include "sse_state.h"

/* Global SSE register file and MXCSR */
XMM_REG  sse_xmm[8];
uint32_t sse_mxcsr = 0x1F80; /* all exceptions masked, round to nearest */

void
sse_init(void)
{
    memset(sse_xmm, 0, sizeof(sse_xmm));
    sse_mxcsr = 0x1F80;
}
