/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          SSE/XMM register state definitions.
 *
 * Authors: frostbite3000
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
#ifndef EMU_SSE_STATE_H
#define EMU_SSE_STATE_H

#include <stdint.h>

/*
 * 128-bit XMM register represented as two 64-bit halves.
 * lo = bits [63:0], hi = bits [127:64].
 */
typedef struct {
    uint64_t lo;
    uint64_t hi;
} XMM_REG;

/* 8 XMM registers (XMM0–XMM7) in 32-bit mode */
extern XMM_REG  sse_xmm[8];

/* MXCSR control/status register */
extern uint32_t sse_mxcsr;

/* Initialise XMM state (zero all regs, set default MXCSR = 0x1F80) */
extern void sse_init(void);

#endif /* EMU_SSE_STATE_H */
