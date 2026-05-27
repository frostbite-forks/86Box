/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          SSE (Streaming SIMD Extensions) instruction emulation.
 *          Implements the interpreter-mode SSE ops that are wired into
 *          the REPE (F3), REPNE (F2), plain 0F, and 66-0F opcode tables.
 *
 *          Only the scalar/packed-single and a handful of management ops
 *          are implemented here; the integer SIMD ops reuse the existing
 *          MMX paths.
 *
 * Authors: frostbite3000
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef X86_OPS_SSE_H
#define X86_OPS_SSE_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include "sse_state.h"

/* -----------------------------------------------------------------------
 * MXCSR bit definitions
 * ----------------------------------------------------------------------- */
#define MXCSR_IE   (1 << 0)   /* Invalid operation flag */
#define MXCSR_DE   (1 << 1)   /* Denormal flag */
#define MXCSR_ZE   (1 << 2)   /* Divide-by-zero flag */
#define MXCSR_OE   (1 << 3)   /* Overflow flag */
#define MXCSR_UE   (1 << 4)   /* Underflow flag */
#define MXCSR_PE   (1 << 5)   /* Precision flag */
#define MXCSR_DAZ  (1 << 6)   /* Denormals are zeros */
#define MXCSR_IM   (1 << 7)   /* Invalid operation mask */
#define MXCSR_DM   (1 << 8)   /* Denormal mask */
#define MXCSR_ZM   (1 << 9)   /* Divide-by-zero mask */
#define MXCSR_OM   (1 << 10)  /* Overflow mask */
#define MXCSR_UM   (1 << 11)  /* Underflow mask */
#define MXCSR_PM   (1 << 12)  /* Precision mask */
#define MXCSR_RC   (3 << 13)  /* Rounding control */
#define MXCSR_FZ   (1 << 15)  /* Flush to zero */
#define MXCSR_RESERVED_MASK 0x0000ffff

/* -----------------------------------------------------------------------
 * Gate macros
 * ----------------------------------------------------------------------- */

/* SSE requires OSFXSR bit in CR4 to be set, and TS/EM bits in CR0 cleared */
#define SSE_ENTER()                                        \
    do {                                                   \
        if (!cpu_has_feature(CPU_FEATURE_SSE)) {           \
            cpu_state.pc = cpu_state.oldpc;                \
            x86illegal();                                  \
            return 1;                                      \
        }                                                  \
        if (cr0 & 0x4) {      /* EM bit */                 \
            cpu_state.pc = cpu_state.oldpc;                \
            x86illegal();                                  \
            return 1;                                      \
        }                                                  \
        if (cr0 & 0x8) {      /* TS bit */                 \
            x86_int(7);                                    \
            return 1;                                      \
        }                                                  \
    } while (0)

/* -----------------------------------------------------------------------
 * Register accessors
 * ----------------------------------------------------------------------- */
#define XMM_GETSRC_a16(s)                                              \
    do {                                                               \
        if (cpu_mod == 3) {                                            \
            (s) = sse_xmm[(cpu_rm) & 7];                               \
            CLOCK_CYCLES(1);                                           \
        } else {                                                       \
            SEG_CHECK_READ(cpu_state.ea_seg);                          \
            (s).lo = readmemq(easeg, cpu_state.eaaddr);               \
            (s).hi = readmemq(easeg, cpu_state.eaaddr + 8);           \
            if (cpu_state.abrt) return 1;                              \
            CLOCK_CYCLES(2);                                           \
        }                                                              \
    } while (0)

#define XMM_GETSRC_a32(s)  XMM_GETSRC_a16(s)

/* Store 128-bit XMM to memory (16-byte aligned required by hardware, not
   checked here – the OS enabling OSFXSR is responsible). */
#define XMM_STOREMEM_a16(v)                                            \
    do {                                                               \
        writememq(easeg, cpu_state.eaaddr,     (v).lo);               \
        writememq(easeg, cpu_state.eaaddr + 8, (v).hi);               \
        if (cpu_state.abrt) return 1;                                  \
    } while (0)
#define XMM_STOREMEM_a32(v)  XMM_STOREMEM_a16(v)

/* -----------------------------------------------------------------------
 * Floating-point helpers
 * ----------------------------------------------------------------------- */
static __inline float
sse_f32(uint32_t bits)
{
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static __inline uint32_t
sse_bits32(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}

static __inline double
sse_f64(uint64_t bits)
{
    double d;
    memcpy(&d, &bits, 8);
    return d;
}

static __inline uint64_t
sse_bits64(double d)
{
    uint64_t b;
    memcpy(&b, &d, 8);
    return b;
}

/* -----------------------------------------------------------------------
 * LDMXCSR / STMXCSR
 * ----------------------------------------------------------------------- */
static int
opLDMXCSR_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_READ(cpu_state.ea_seg);
    sse_mxcsr = readmeml(easeg, cpu_state.eaaddr) & MXCSR_RESERVED_MASK;
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(4);
    return 0;
}
static int
opLDMXCSR_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_READ(cpu_state.ea_seg);
    sse_mxcsr = readmeml(easeg, cpu_state.eaaddr) & MXCSR_RESERVED_MASK;
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(4);
    return 0;
}

static int
opSTMXCSR_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememl(easeg, cpu_state.eaaddr, sse_mxcsr);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(4);
    return 0;
}
static int
opSTMXCSR_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememl(easeg, cpu_state.eaaddr, sse_mxcsr);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(4);
    return 0;
}

/* -----------------------------------------------------------------------
 * SFENCE (0F AE /7)
 * ----------------------------------------------------------------------- */
static int
opSFENCE(UNUSED(uint32_t fetchdat))
{
    /* On an emulator there is no store buffer to flush, so this is a NOP. */
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * 0F AE dispatcher (FXSAVE/FXRSTOR handled elsewhere, LDMXCSR/STMXCSR/
 * SFENCE handled here)
 * ----------------------------------------------------------------------- */
static int
opSSE_AE_a16(uint32_t fetchdat)
{
    uint8_t mod_rm = fetchdat & 0xff;
    uint8_t sub    = (mod_rm >> 3) & 7;
    uint8_t rm     = mod_rm & 7;
    uint8_t mod    = (mod_rm >> 6) & 3;

    SSE_ENTER();

    switch (sub) {
        case 2: /* LDMXCSR – mem only */
            return opLDMXCSR_a16(fetchdat);
        case 3: /* STMXCSR – mem only */
            return opSTMXCSR_a16(fetchdat);
        case 7: /* SFENCE (mod=3) or CLFLUSH (mod!=3, optional) */
            if (mod == 3)
                return opSFENCE(fetchdat);
            /* CLFLUSH: just decode, treat as NOP */
            fetch_ea_16(fetchdat);
            CLOCK_CYCLES(1);
            return 0;
        default:
            /* Pass through to FXSAVE/FXRSTOR (sub 0/1) handled upstream */
            cpu_state.pc = cpu_state.oldpc;
            x86illegal();
            return 1;
    }
}

static int
opSSE_AE_a32(uint32_t fetchdat)
{
    uint8_t mod_rm = fetchdat & 0xff;
    uint8_t sub    = (mod_rm >> 3) & 7;
    uint8_t mod    = (mod_rm >> 6) & 3;

    SSE_ENTER();

    switch (sub) {
        case 2:
            return opLDMXCSR_a32(fetchdat);
        case 3:
            return opSTMXCSR_a32(fetchdat);
        case 7:
            if (mod == 3)
                return opSFENCE(fetchdat);
            fetch_ea_32(fetchdat);
            CLOCK_CYCLES(1);
            return 0;
        default:
            cpu_state.pc = cpu_state.oldpc;
            x86illegal();
            return 1;
    }
}

/* -----------------------------------------------------------------------
 * MOVAPS/MOVUPS  (0F 28, 0F 29, 0F 10, 0F 11)
 * ----------------------------------------------------------------------- */
#define DEF_MOVAPS_LOAD(suffix, bits)                                  \
static int                                                             \
opMOVAPS_xmm_xmm_##suffix(uint32_t fetchdat)                          \
{                                                                      \
    XMM_REG src;                                                       \
    SSE_ENTER();                                                       \
    fetch_ea_##bits(fetchdat);                                         \
    XMM_GETSRC_##suffix(src);                                          \
    sse_xmm[cpu_reg & 7] = src;                                        \
    CLOCK_CYCLES(1);                                                   \
    return 0;                                                          \
}

#define DEF_MOVAPS_STORE(suffix, bits)                                 \
static int                                                             \
opMOVAPS_xmm_store_##suffix(uint32_t fetchdat)                        \
{                                                                      \
    SSE_ENTER();                                                       \
    fetch_ea_##bits(fetchdat);                                         \
    if (cpu_mod == 3) {                                                \
        sse_xmm[cpu_rm & 7] = sse_xmm[cpu_reg & 7];                   \
    } else {                                                           \
        SEG_CHECK_WRITE(cpu_state.ea_seg);                             \
        XMM_STOREMEM_##suffix(sse_xmm[cpu_reg & 7]);                  \
    }                                                                  \
    CLOCK_CYCLES(1);                                                   \
    return 0;                                                          \
}

DEF_MOVAPS_LOAD(a16, 16)
DEF_MOVAPS_LOAD(a32, 32)
DEF_MOVAPS_STORE(a16, 16)
DEF_MOVAPS_STORE(a32, 32)

/* MOVSS (F3 0F 10 / F3 0F 11) – scalar single */
static int
opMOVSS_load_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    if (cpu_mod == 3) {
        /* xmm_dst[31:0] = xmm_src[31:0]; upper bits unchanged */
        sse_xmm[cpu_reg & 7].lo =
            (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) |
            (sse_xmm[cpu_rm & 7].lo  & 0x00000000FFFFFFFFULL);
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t v = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        /* When loading from memory zero-extend to 128 bits */
        sse_xmm[cpu_reg & 7].lo = (uint64_t)v;
        sse_xmm[cpu_reg & 7].hi = 0;
    }
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVSS_load_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    if (cpu_mod == 3) {
        sse_xmm[cpu_reg & 7].lo =
            (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) |
            (sse_xmm[cpu_rm & 7].lo  & 0x00000000FFFFFFFFULL);
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t v = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        sse_xmm[cpu_reg & 7].lo = (uint64_t)v;
        sse_xmm[cpu_reg & 7].hi = 0;
    }
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVSS_store_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    if (cpu_mod == 3) {
        uint32_t v = (uint32_t)(sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF);
        sse_xmm[cpu_rm & 7].lo =
            (sse_xmm[cpu_rm & 7].lo & 0xFFFFFFFF00000000ULL) | v;
    } else {
        SEG_CHECK_WRITE(cpu_state.ea_seg);
        writememl(easeg, cpu_state.eaaddr,
                  (uint32_t)(sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF));
        if (cpu_state.abrt) return 1;
    }
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVSS_store_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    if (cpu_mod == 3) {
        uint32_t v = (uint32_t)(sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF);
        sse_xmm[cpu_rm & 7].lo =
            (sse_xmm[cpu_rm & 7].lo & 0xFFFFFFFF00000000ULL) | v;
    } else {
        SEG_CHECK_WRITE(cpu_state.ea_seg);
        writememl(easeg, cpu_state.eaaddr,
                  (uint32_t)(sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF));
        if (cpu_state.abrt) return 1;
    }
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * Scalar single-precision arithmetic  (F3 0F)
 * -----------------------------------------------------------------------
 * Macro to generate ADDSS/SUBSS/MULSS/DIVSS/etc.
 */
#define DEF_SSE_SCALAR_OP_SS(name, op, cycles)                        \
static int                                                             \
op##name##SS_a16(uint32_t fetchdat)                                   \
{                                                                      \
    SSE_ENTER();                                                       \
    fetch_ea_16(fetchdat);                                             \
    float dst_f, src_f;                                                \
    uint32_t dst_bits = (uint32_t)(sse_xmm[cpu_reg & 7].lo);         \
    if (cpu_mod == 3) {                                                \
        uint32_t src_bits = (uint32_t)(sse_xmm[cpu_rm & 7].lo);      \
        dst_f = sse_f32(dst_bits); src_f = sse_f32(src_bits);        \
    } else {                                                           \
        SEG_CHECK_READ(cpu_state.ea_seg);                              \
        uint32_t src_bits = readmeml(easeg, cpu_state.eaaddr);        \
        if (cpu_state.abrt) return 1;                                  \
        dst_f = sse_f32(dst_bits); src_f = sse_f32(src_bits);        \
    }                                                                  \
    float res = dst_f op src_f;                                        \
    sse_xmm[cpu_reg & 7].lo =                                         \
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) |           \
        sse_bits32(res);                                               \
    CLOCK_CYCLES(cycles);                                              \
    return 0;                                                          \
}                                                                      \
static int                                                             \
op##name##SS_a32(uint32_t fetchdat)                                   \
{                                                                      \
    SSE_ENTER();                                                       \
    fetch_ea_32(fetchdat);                                             \
    float dst_f, src_f;                                                \
    uint32_t dst_bits = (uint32_t)(sse_xmm[cpu_reg & 7].lo);         \
    if (cpu_mod == 3) {                                                \
        uint32_t src_bits = (uint32_t)(sse_xmm[cpu_rm & 7].lo);      \
        dst_f = sse_f32(dst_bits); src_f = sse_f32(src_bits);        \
    } else {                                                           \
        SEG_CHECK_READ(cpu_state.ea_seg);                              \
        uint32_t src_bits = readmeml(easeg, cpu_state.eaaddr);        \
        if (cpu_state.abrt) return 1;                                  \
        dst_f = sse_f32(dst_bits); src_f = sse_f32(src_bits);        \
    }                                                                  \
    float res = dst_f op src_f;                                        \
    sse_xmm[cpu_reg & 7].lo =                                         \
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) |           \
        sse_bits32(res);                                               \
    CLOCK_CYCLES(cycles);                                              \
    return 0;                                                          \
}

DEF_SSE_SCALAR_OP_SS(ADD, +, 3)
DEF_SSE_SCALAR_OP_SS(SUB, -, 3)
DEF_SSE_SCALAR_OP_SS(MUL, *, 4)
DEF_SSE_SCALAR_OP_SS(DIV, /, 17)

/* SQRTSS */
static int
opSQRTSS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float res = sqrtf(src_f);
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(20);
    return 0;
}
static int
opSQRTSS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float res = sqrtf(src_f);
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(20);
    return 0;
}

/* RSQRTSS */
static int
opRSQRTSS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float res = 1.0f / sqrtf(src_f);
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opRSQRTSS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float res = 1.0f / sqrtf(src_f);
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(4);
    return 0;
}

/* RCPSS */
static int
opRCPSS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float res = 1.0f / src_f;
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opRCPSS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float res = 1.0f / src_f;
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(4);
    return 0;
}

/* -----------------------------------------------------------------------
 * Packed single-precision arithmetic  (0F, no prefix)
 * -----------------------------------------------------------------------
 * Each XMM register holds 4 × float32.
 */
#define PS_LANE0(r)  sse_f32((uint32_t)((r).lo))
#define PS_LANE1(r)  sse_f32((uint32_t)((r).lo >> 32))
#define PS_LANE2(r)  sse_f32((uint32_t)((r).hi))
#define PS_LANE3(r)  sse_f32((uint32_t)((r).hi >> 32))

#define PS_PACK(l0, l1, l2, l3) \
    { .lo = ((uint64_t)sse_bits32(l0) | ((uint64_t)sse_bits32(l1) << 32)), \
      .hi = ((uint64_t)sse_bits32(l2) | ((uint64_t)sse_bits32(l3) << 32)) }

#define DEF_SSE_PACKED_OP_PS(name, op, cycles)                         \
static int                                                              \
op##name##PS_a16(uint32_t fetchdat)                                    \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_16(fetchdat);                                              \
    XMM_GETSRC_a16(src);                                               \
    XMM_REG *d = &sse_xmm[cpu_reg & 7];                               \
    float r0 = PS_LANE0(*d) op PS_LANE0(src);                          \
    float r1 = PS_LANE1(*d) op PS_LANE1(src);                          \
    float r2 = PS_LANE2(*d) op PS_LANE2(src);                          \
    float r3 = PS_LANE3(*d) op PS_LANE3(src);                          \
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32); \
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32); \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}                                                                       \
static int                                                              \
op##name##PS_a32(uint32_t fetchdat)                                    \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_32(fetchdat);                                              \
    XMM_GETSRC_a32(src);                                               \
    XMM_REG *d = &sse_xmm[cpu_reg & 7];                               \
    float r0 = PS_LANE0(*d) op PS_LANE0(src);                          \
    float r1 = PS_LANE1(*d) op PS_LANE1(src);                          \
    float r2 = PS_LANE2(*d) op PS_LANE2(src);                          \
    float r3 = PS_LANE3(*d) op PS_LANE3(src);                          \
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32); \
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32); \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}

DEF_SSE_PACKED_OP_PS(ADD, +, 3)
DEF_SSE_PACKED_OP_PS(SUB, -, 3)
DEF_SSE_PACKED_OP_PS(MUL, *, 4)
DEF_SSE_PACKED_OP_PS(DIV, /, 17)

/* SQRTPS */
#define DEF_SSE_SQRT_PS(suffix, bits)                                   \
static int                                                              \
opSQRTPS_##suffix(uint32_t fetchdat)                                   \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_##bits(fetchdat);                                          \
    XMM_GETSRC_##suffix(src);                                          \
    XMM_REG *d = &sse_xmm[cpu_reg & 7];                               \
    float r0 = sqrtf(PS_LANE0(src));                                    \
    float r1 = sqrtf(PS_LANE1(src));                                    \
    float r2 = sqrtf(PS_LANE2(src));                                    \
    float r3 = sqrtf(PS_LANE3(src));                                    \
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32); \
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32); \
    CLOCK_CYCLES(56);                                                   \
    return 0;                                                           \
}
DEF_SSE_SQRT_PS(a16, 16)
DEF_SSE_SQRT_PS(a32, 32)

/* RSQRTPS */
#define DEF_SSE_RSQRT_PS(suffix, bits)                                  \
static int                                                              \
opRSQRTPS_##suffix(uint32_t fetchdat)                                  \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_##bits(fetchdat);                                          \
    XMM_GETSRC_##suffix(src);                                          \
    XMM_REG *d = &sse_xmm[cpu_reg & 7];                               \
    float r0 = 1.0f / sqrtf(PS_LANE0(src));                            \
    float r1 = 1.0f / sqrtf(PS_LANE1(src));                            \
    float r2 = 1.0f / sqrtf(PS_LANE2(src));                            \
    float r3 = 1.0f / sqrtf(PS_LANE3(src));                            \
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32); \
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32); \
    CLOCK_CYCLES(8);                                                    \
    return 0;                                                           \
}
DEF_SSE_RSQRT_PS(a16, 16)
DEF_SSE_RSQRT_PS(a32, 32)

/* RCPPS */
#define DEF_SSE_RCP_PS(suffix, bits)                                    \
static int                                                              \
opRCPPS_##suffix(uint32_t fetchdat)                                    \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_##bits(fetchdat);                                          \
    XMM_GETSRC_##suffix(src);                                          \
    XMM_REG *d = &sse_xmm[cpu_reg & 7];                               \
    float r0 = 1.0f / PS_LANE0(src);                                   \
    float r1 = 1.0f / PS_LANE1(src);                                   \
    float r2 = 1.0f / PS_LANE2(src);                                   \
    float r3 = 1.0f / PS_LANE3(src);                                   \
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32); \
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32); \
    CLOCK_CYCLES(8);                                                    \
    return 0;                                                           \
}
DEF_SSE_RCP_PS(a16, 16)
DEF_SSE_RCP_PS(a32, 32)

/* -----------------------------------------------------------------------
 * MAXPS / MINPS / MAXSS / MINSS
 * ----------------------------------------------------------------------- */
#define DEF_SSE_MINMAX_PS(name, op, cycles)                             \
static int                                                              \
op##name##PS_a16(uint32_t fetchdat)                                    \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_16(fetchdat);                                              \
    XMM_GETSRC_a16(src);                                               \
    XMM_REG *d = &sse_xmm[cpu_reg & 7];                               \
    float r0 = (PS_LANE0(*d) op PS_LANE0(src)) ? PS_LANE0(*d) : PS_LANE0(src); \
    float r1 = (PS_LANE1(*d) op PS_LANE1(src)) ? PS_LANE1(*d) : PS_LANE1(src); \
    float r2 = (PS_LANE2(*d) op PS_LANE2(src)) ? PS_LANE2(*d) : PS_LANE2(src); \
    float r3 = (PS_LANE3(*d) op PS_LANE3(src)) ? PS_LANE3(*d) : PS_LANE3(src); \
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32); \
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32); \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}                                                                       \
static int                                                              \
op##name##PS_a32(uint32_t fetchdat)                                    \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_32(fetchdat);                                              \
    XMM_GETSRC_a32(src);                                               \
    XMM_REG *d = &sse_xmm[cpu_reg & 7];                               \
    float r0 = (PS_LANE0(*d) op PS_LANE0(src)) ? PS_LANE0(*d) : PS_LANE0(src); \
    float r1 = (PS_LANE1(*d) op PS_LANE1(src)) ? PS_LANE1(*d) : PS_LANE1(src); \
    float r2 = (PS_LANE2(*d) op PS_LANE2(src)) ? PS_LANE2(*d) : PS_LANE2(src); \
    float r3 = (PS_LANE3(*d) op PS_LANE3(src)) ? PS_LANE3(*d) : PS_LANE3(src); \
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32); \
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32); \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}

DEF_SSE_MINMAX_PS(MIN, <, 3)
DEF_SSE_MINMAX_PS(MAX, >, 3)

/* Scalar variants */
#define DEF_SSE_MINMAX_SS(name, op, cycles)                             \
static int                                                              \
op##name##SS_a16(uint32_t fetchdat)                                    \
{                                                                       \
    SSE_ENTER();                                                        \
    fetch_ea_16(fetchdat);                                              \
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));       \
    float src_f;                                                        \
    if (cpu_mod == 3)                                                   \
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));          \
    else {                                                              \
        SEG_CHECK_READ(cpu_state.ea_seg);                               \
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);               \
        if (cpu_state.abrt) return 1;                                   \
        src_f = sse_f32(sb);                                           \
    }                                                                   \
    float res = (dst_f op src_f) ? dst_f : src_f;                      \
    sse_xmm[cpu_reg & 7].lo =                                          \
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res); \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}                                                                       \
static int                                                              \
op##name##SS_a32(uint32_t fetchdat)                                    \
{                                                                       \
    SSE_ENTER();                                                        \
    fetch_ea_32(fetchdat);                                              \
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));       \
    float src_f;                                                        \
    if (cpu_mod == 3)                                                   \
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));          \
    else {                                                              \
        SEG_CHECK_READ(cpu_state.ea_seg);                               \
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);               \
        if (cpu_state.abrt) return 1;                                   \
        src_f = sse_f32(sb);                                           \
    }                                                                   \
    float res = (dst_f op src_f) ? dst_f : src_f;                      \
    sse_xmm[cpu_reg & 7].lo =                                          \
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res); \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}

DEF_SSE_MINMAX_SS(MIN, <, 3)
DEF_SSE_MINMAX_SS(MAX, >, 3)

/* -----------------------------------------------------------------------
 * ANDPS / ANDNPS / ORPS / XORPS
 * ----------------------------------------------------------------------- */
#define DEF_SSE_BITWISE_PS(name, op, cycles)                            \
static int                                                              \
op##name##PS_a16(uint32_t fetchdat)                                    \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_16(fetchdat);                                              \
    XMM_GETSRC_a16(src);                                               \
    sse_xmm[cpu_reg & 7].lo op##= src.lo;                              \
    sse_xmm[cpu_reg & 7].hi op##= src.hi;                              \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}                                                                       \
static int                                                              \
op##name##PS_a32(uint32_t fetchdat)                                    \
{                                                                       \
    XMM_REG src;                                                        \
    SSE_ENTER();                                                        \
    fetch_ea_32(fetchdat);                                              \
    XMM_GETSRC_a32(src);                                               \
    sse_xmm[cpu_reg & 7].lo op##= src.lo;                              \
    sse_xmm[cpu_reg & 7].hi op##= src.hi;                              \
    CLOCK_CYCLES(cycles);                                               \
    return 0;                                                           \
}

DEF_SSE_BITWISE_PS(AND,  &, 1)
DEF_SSE_BITWISE_PS(OR,   |, 1)
DEF_SSE_BITWISE_PS(XOR,  ^, 1)

/* ANDNPS  (d = ~d & src) */
static int
opANDNPS_a16(uint32_t fetchdat)
{
    XMM_REG src;
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    XMM_GETSRC_a16(src);
    sse_xmm[cpu_reg & 7].lo = (~sse_xmm[cpu_reg & 7].lo) & src.lo;
    sse_xmm[cpu_reg & 7].hi = (~sse_xmm[cpu_reg & 7].hi) & src.hi;
    CLOCK_CYCLES(1);
    return 0;
}
static int
opANDNPS_a32(uint32_t fetchdat)
{
    XMM_REG src;
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    XMM_GETSRC_a32(src);
    sse_xmm[cpu_reg & 7].lo = (~sse_xmm[cpu_reg & 7].lo) & src.lo;
    sse_xmm[cpu_reg & 7].hi = (~sse_xmm[cpu_reg & 7].hi) & src.hi;
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * CMPPS / CMPSS
 * Comparison predicates (imm8):
 *   0=EQ  1=LT  2=LE  3=UNORD  4=NEQ  5=NLT  6=NLE  7=ORD
 * ----------------------------------------------------------------------- */
static __inline uint32_t
sse_cmp_f32(float a, float b, uint8_t pred)
{
    int res;
    switch (pred & 7) {
        case 0: res = (a == b);  break;
        case 1: res = (a < b);   break;
        case 2: res = (a <= b);  break;
        case 3: res = (isnan(a) || isnan(b)); break;
        case 4: res = (a != b);  break;
        case 5: res = !(a < b);  break;
        case 6: res = !(a <= b); break;
        case 7: res = !(isnan(a) || isnan(b)); break;
        default: res = 0; break;
    }
    return res ? 0xFFFFFFFFU : 0;
}

static int
opCMPPS_a16(uint32_t fetchdat)
{
    XMM_REG src;
    uint8_t  pred;
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    XMM_GETSRC_a16(src);
    pred = (uint8_t)readmemb(cs, cpu_state.pc++);
    if (cpu_state.abrt) return 1;
    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    uint32_t r0 = sse_cmp_f32(PS_LANE0(*d), PS_LANE0(src), pred);
    uint32_t r1 = sse_cmp_f32(PS_LANE1(*d), PS_LANE1(src), pred);
    uint32_t r2 = sse_cmp_f32(PS_LANE2(*d), PS_LANE2(src), pred);
    uint32_t r3 = sse_cmp_f32(PS_LANE3(*d), PS_LANE3(src), pred);
    d->lo = ((uint64_t)r0) | ((uint64_t)r1 << 32);
    d->hi = ((uint64_t)r2) | ((uint64_t)r3 << 32);
    CLOCK_CYCLES(2);
    return 0;
}
static int
opCMPPS_a32(uint32_t fetchdat)
{
    XMM_REG src;
    uint8_t  pred;
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    XMM_GETSRC_a32(src);
    pred = (uint8_t)readmemb(cs, cpu_state.pc++);
    if (cpu_state.abrt) return 1;
    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    uint32_t r0 = sse_cmp_f32(PS_LANE0(*d), PS_LANE0(src), pred);
    uint32_t r1 = sse_cmp_f32(PS_LANE1(*d), PS_LANE1(src), pred);
    uint32_t r2 = sse_cmp_f32(PS_LANE2(*d), PS_LANE2(src), pred);
    uint32_t r3 = sse_cmp_f32(PS_LANE3(*d), PS_LANE3(src), pred);
    d->lo = ((uint64_t)r0) | ((uint64_t)r1 << 32);
    d->hi = ((uint64_t)r2) | ((uint64_t)r3 << 32);
    CLOCK_CYCLES(2);
    return 0;
}

/* CMPSS (F3 prefix) */
static int
opCMPSS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    uint8_t pred = (uint8_t)readmemb(cs, cpu_state.pc++);
    if (cpu_state.abrt) return 1;
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));
    uint32_t res = sse_cmp_f32(dst_f, src_f, pred);
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | res;
    CLOCK_CYCLES(2);
    return 0;
}
static int
opCMPSS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    uint8_t pred = (uint8_t)readmemb(cs, cpu_state.pc++);
    if (cpu_state.abrt) return 1;
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));
    uint32_t res = sse_cmp_f32(dst_f, src_f, pred);
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | res;
    CLOCK_CYCLES(2);
    return 0;
}

/* -----------------------------------------------------------------------
 * COMISS / UCOMISS  (0F 2F / 0F 2E)
 * Sets EFLAGS ZF/PF/CF from ordered/unordered comparison.
 * ----------------------------------------------------------------------- */
static __inline void
sse_comiss_flags(float a, float b, int unordered_ok)
{
    /* Clear CF, PF, ZF, OF, SF, AF */
    cpu_state.flags &= ~(0x8d5);
    if (isnan(a) || isnan(b)) {
        if (!unordered_ok) {
            /* Invalid operation – set IE in MXCSR if unmasked */
            sse_mxcsr |= MXCSR_IE;
        }
        /* Unordered: ZF=1, PF=1, CF=1 */
        cpu_state.flags |= 0x45; /* CF|ZF and later PF */
        cpu_state.eflags |= 4;   /* PF in eflags high */
    } else if (a < b) {
        /* Below: CF=1, ZF=0, PF=0 */
        cpu_state.flags |= 1;  /* CF */
    } else if (a > b) {
        /* Above: CF=0, ZF=0, PF=0 – already clear */
    } else {
        /* Equal: ZF=1, CF=0, PF=0 */
        cpu_state.flags |= 0x40; /* ZF */
    }
}

static int
opCOMISS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));
    sse_comiss_flags(dst_f, src_f, 0);
    CLOCK_CYCLES(1);
    return 0;
}
static int
opCOMISS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));
    sse_comiss_flags(dst_f, src_f, 0);
    CLOCK_CYCLES(1);
    return 0;
}

static int
opUCOMISS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));
    sse_comiss_flags(dst_f, src_f, 1);
    CLOCK_CYCLES(1);
    return 0;
}
static int
opUCOMISS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    float dst_f = sse_f32((uint32_t)(sse_xmm[cpu_reg & 7].lo));
    sse_comiss_flags(dst_f, src_f, 1);
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * CVTSI2SS / CVTSS2SI / CVTTSS2SI  (F3 0F 2A / 2D / 2C)
 * ----------------------------------------------------------------------- */
static int
opCVTSI2SS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    int32_t src_i;
    if (cpu_mod == 3) {
        src_i = (int32_t)cpu_state.regs[cpu_rm & 7].l;
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src_i = (int32_t)readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    float res = (float)src_i;
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opCVTSI2SS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    int32_t src_i;
    if (cpu_mod == 3) {
        src_i = (int32_t)cpu_state.regs[cpu_rm & 7].l;
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src_i = (int32_t)readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    float res = (float)src_i;
    sse_xmm[cpu_reg & 7].lo =
        (sse_xmm[cpu_reg & 7].lo & 0xFFFFFFFF00000000ULL) | sse_bits32(res);
    CLOCK_CYCLES(4);
    return 0;
}

static int
opCVTSS2SI_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    cpu_state.regs[cpu_reg & 7].l = (uint32_t)(int32_t)nearbyintf(src_f);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opCVTSS2SI_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    cpu_state.regs[cpu_reg & 7].l = (uint32_t)(int32_t)nearbyintf(src_f);
    CLOCK_CYCLES(4);
    return 0;
}

static int
opCVTTSS2SI_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    cpu_state.regs[cpu_reg & 7].l = (uint32_t)(int32_t)truncf(src_f);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opCVTTSS2SI_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src_f;
    if (cpu_mod == 3) {
        src_f = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        uint32_t sb = readmeml(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
        src_f = sse_f32(sb);
    }
    cpu_state.regs[cpu_reg & 7].l = (uint32_t)(int32_t)truncf(src_f);
    CLOCK_CYCLES(4);
    return 0;
}

/* -----------------------------------------------------------------------
 * CVTPS2PI / CVTTPS2PI / CVTPI2PS  (0F 2A, 2C, 2D)
 * These convert between XMM packed-float and MMX 64-bit int.
 * ----------------------------------------------------------------------- */
static int
opCVTPI2PS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    MMX_REG src;
    if (cpu_mod == 3) {
        src = MMX_GETREG(cpu_rm);
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src.q = readmemq(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    float f0 = (float)(int32_t)src.l[0];
    float f1 = (float)(int32_t)src.l[1];
    sse_xmm[cpu_reg & 7].lo =
        (uint64_t)sse_bits32(f0) | ((uint64_t)sse_bits32(f1) << 32);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opCVTPI2PS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    MMX_REG src;
    if (cpu_mod == 3) {
        src = MMX_GETREG(cpu_rm);
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src.q = readmemq(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    float f0 = (float)(int32_t)src.l[0];
    float f1 = (float)(int32_t)src.l[1];
    sse_xmm[cpu_reg & 7].lo =
        (uint64_t)sse_bits32(f0) | ((uint64_t)sse_bits32(f1) << 32);
    CLOCK_CYCLES(4);
    return 0;
}

static int
opCVTPS2PI_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src0, src1;
    if (cpu_mod == 3) {
        src0 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
        src1 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo >> 32));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src0 = sse_f32(readmeml(easeg, cpu_state.eaaddr));
        src1 = sse_f32(readmeml(easeg, cpu_state.eaaddr + 4));
        if (cpu_state.abrt) return 1;
    }
    MMX_REG *dst = MMX_GETREGP(cpu_reg);
    dst->l[0] = (uint32_t)(int32_t)nearbyintf(src0);
    dst->l[1] = (uint32_t)(int32_t)nearbyintf(src1);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opCVTPS2PI_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src0, src1;
    if (cpu_mod == 3) {
        src0 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
        src1 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo >> 32));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src0 = sse_f32(readmeml(easeg, cpu_state.eaaddr));
        src1 = sse_f32(readmeml(easeg, cpu_state.eaaddr + 4));
        if (cpu_state.abrt) return 1;
    }
    MMX_REG *dst = MMX_GETREGP(cpu_reg);
    dst->l[0] = (uint32_t)(int32_t)nearbyintf(src0);
    dst->l[1] = (uint32_t)(int32_t)nearbyintf(src1);
    CLOCK_CYCLES(4);
    return 0;
}

static int
opCVTTPS2PI_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    float src0, src1;
    if (cpu_mod == 3) {
        src0 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
        src1 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo >> 32));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src0 = sse_f32(readmeml(easeg, cpu_state.eaaddr));
        src1 = sse_f32(readmeml(easeg, cpu_state.eaaddr + 4));
        if (cpu_state.abrt) return 1;
    }
    MMX_REG *dst = MMX_GETREGP(cpu_reg);
    dst->l[0] = (uint32_t)(int32_t)truncf(src0);
    dst->l[1] = (uint32_t)(int32_t)truncf(src1);
    CLOCK_CYCLES(4);
    return 0;
}
static int
opCVTTPS2PI_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    float src0, src1;
    if (cpu_mod == 3) {
        src0 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo));
        src1 = sse_f32((uint32_t)(sse_xmm[cpu_rm & 7].lo >> 32));
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        src0 = sse_f32(readmeml(easeg, cpu_state.eaaddr));
        src1 = sse_f32(readmeml(easeg, cpu_state.eaaddr + 4));
        if (cpu_state.abrt) return 1;
    }
    MMX_REG *dst = MMX_GETREGP(cpu_reg);
    dst->l[0] = (uint32_t)(int32_t)truncf(src0);
    dst->l[1] = (uint32_t)(int32_t)truncf(src1);
    CLOCK_CYCLES(4);
    return 0;
}

/* -----------------------------------------------------------------------
 * UNPCKLPS / UNPCKHPS  (0F 14 / 0F 15)
 * ----------------------------------------------------------------------- */
static int
opUNPCKLPS_a16(uint32_t fetchdat)
{
    XMM_REG src;
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    XMM_GETSRC_a16(src);
    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    /* dst[127:64] = src[63:32] : dst[63:32]  */
    /* dst[63:0]   = src[31:0]  : dst[31:0]   */
    uint32_t d0 = (uint32_t)d->lo;
    uint32_t d1 = (uint32_t)(d->lo >> 32);
    uint32_t s0 = (uint32_t)src.lo;
    uint32_t s1 = (uint32_t)(src.lo >> 32);
    d->lo = (uint64_t)d0 | ((uint64_t)s0 << 32);
    d->hi = (uint64_t)d1 | ((uint64_t)s1 << 32);
    CLOCK_CYCLES(1);
    return 0;
}
static int
opUNPCKLPS_a32(uint32_t fetchdat)
{
    XMM_REG src;
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    XMM_GETSRC_a32(src);
    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    uint32_t d0 = (uint32_t)d->lo;
    uint32_t d1 = (uint32_t)(d->lo >> 32);
    uint32_t s0 = (uint32_t)src.lo;
    uint32_t s1 = (uint32_t)(src.lo >> 32);
    d->lo = (uint64_t)d0 | ((uint64_t)s0 << 32);
    d->hi = (uint64_t)d1 | ((uint64_t)s1 << 32);
    CLOCK_CYCLES(1);
    return 0;
}

static int
opUNPCKHPS_a16(uint32_t fetchdat)
{
    XMM_REG src;
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    XMM_GETSRC_a16(src);
    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    uint32_t d2 = (uint32_t)d->hi;
    uint32_t d3 = (uint32_t)(d->hi >> 32);
    uint32_t s2 = (uint32_t)src.hi;
    uint32_t s3 = (uint32_t)(src.hi >> 32);
    d->lo = (uint64_t)d2 | ((uint64_t)s2 << 32);
    d->hi = (uint64_t)d3 | ((uint64_t)s3 << 32);
    CLOCK_CYCLES(1);
    return 0;
}
static int
opUNPCKHPS_a32(uint32_t fetchdat)
{
    XMM_REG src;
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    XMM_GETSRC_a32(src);
    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    uint32_t d2 = (uint32_t)d->hi;
    uint32_t d3 = (uint32_t)(d->hi >> 32);
    uint32_t s2 = (uint32_t)src.hi;
    uint32_t s3 = (uint32_t)(src.hi >> 32);
    d->lo = (uint64_t)d2 | ((uint64_t)s2 << 32);
    d->hi = (uint64_t)d3 | ((uint64_t)s3 << 32);
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * SHUFPS  (0F C6 /r imm8)
 * ----------------------------------------------------------------------- */
static int
opSHUFPS_a16(uint32_t fetchdat)
{
    XMM_REG src;
    uint8_t  imm;
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    XMM_GETSRC_a16(src);
    imm = (uint8_t)readmemb(cs, cpu_state.pc++);
    if (cpu_state.abrt) return 1;

    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    float dst_lanes[4] = { PS_LANE0(*d), PS_LANE1(*d), PS_LANE2(*d), PS_LANE3(*d) };
    float src_lanes[4] = { PS_LANE0(src), PS_LANE1(src), PS_LANE2(src), PS_LANE3(src) };
    float r0 = dst_lanes[(imm >> 0) & 3];
    float r1 = dst_lanes[(imm >> 2) & 3];
    float r2 = src_lanes[(imm >> 4) & 3];
    float r3 = src_lanes[(imm >> 6) & 3];
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32);
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32);
    CLOCK_CYCLES(1);
    return 0;
}
static int
opSHUFPS_a32(uint32_t fetchdat)
{
    XMM_REG src;
    uint8_t  imm;
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    XMM_GETSRC_a32(src);
    imm = (uint8_t)readmemb(cs, cpu_state.pc++);
    if (cpu_state.abrt) return 1;

    XMM_REG *d = &sse_xmm[cpu_reg & 7];
    float dst_lanes[4] = { PS_LANE0(*d), PS_LANE1(*d), PS_LANE2(*d), PS_LANE3(*d) };
    float src_lanes[4] = { PS_LANE0(src), PS_LANE1(src), PS_LANE2(src), PS_LANE3(src) };
    float r0 = dst_lanes[(imm >> 0) & 3];
    float r1 = dst_lanes[(imm >> 2) & 3];
    float r2 = src_lanes[(imm >> 4) & 3];
    float r3 = src_lanes[(imm >> 6) & 3];
    d->lo = ((uint64_t)sse_bits32(r0)) | ((uint64_t)sse_bits32(r1) << 32);
    d->hi = ((uint64_t)sse_bits32(r2)) | ((uint64_t)sse_bits32(r3) << 32);
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * MOVHLPS (0F 12) / MOVLHPS (0F 16)
 * ----------------------------------------------------------------------- */
static int
opMOVHLPS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    if (cpu_mod == 3) {
        /* dst_lo = src_hi */
        sse_xmm[cpu_reg & 7].lo = sse_xmm[cpu_rm & 7].hi;
    } else {
        /* MOVLPS: load 64 bits into low half */
        SEG_CHECK_READ(cpu_state.ea_seg);
        sse_xmm[cpu_reg & 7].lo = readmemq(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVHLPS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    if (cpu_mod == 3) {
        sse_xmm[cpu_reg & 7].lo = sse_xmm[cpu_rm & 7].hi;
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        sse_xmm[cpu_reg & 7].lo = readmemq(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    CLOCK_CYCLES(1);
    return 0;
}

static int
opMOVLHPS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    if (cpu_mod == 3) {
        /* dst_hi = src_lo */
        sse_xmm[cpu_reg & 7].hi = sse_xmm[cpu_rm & 7].lo;
    } else {
        /* MOVHPS: load 64 bits into high half */
        SEG_CHECK_READ(cpu_state.ea_seg);
        sse_xmm[cpu_reg & 7].hi = readmemq(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVLHPS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    if (cpu_mod == 3) {
        sse_xmm[cpu_reg & 7].hi = sse_xmm[cpu_rm & 7].lo;
    } else {
        SEG_CHECK_READ(cpu_state.ea_seg);
        sse_xmm[cpu_reg & 7].hi = readmemq(easeg, cpu_state.eaaddr);
        if (cpu_state.abrt) return 1;
    }
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * MOVLPS store (0F 13) / MOVHPS store (0F 17)
 * ----------------------------------------------------------------------- */
static int
opMOVLPS_store_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememq(easeg, cpu_state.eaaddr, sse_xmm[cpu_reg & 7].lo);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVLPS_store_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememq(easeg, cpu_state.eaaddr, sse_xmm[cpu_reg & 7].lo);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(1);
    return 0;
}

static int
opMOVHPS_store_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememq(easeg, cpu_state.eaaddr, sse_xmm[cpu_reg & 7].hi);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVHPS_store_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememq(easeg, cpu_state.eaaddr, sse_xmm[cpu_reg & 7].hi);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * MOVMSKPS  (0F 50)  –  move sign bits of 4 packed floats to int reg
 * ----------------------------------------------------------------------- */
static int
opMOVMSKPS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    ILLEGAL_ON(cpu_mod != 3);
    XMM_REG *s = &sse_xmm[cpu_rm & 7];
    uint32_t mask = 0;
    if (s->lo & 0x80000000ULL)        mask |= 1;
    if (s->lo & 0x8000000000000000ULL) mask |= 2;
    if (s->hi & 0x80000000ULL)        mask |= 4;
    if (s->hi & 0x8000000000000000ULL) mask |= 8;
    cpu_state.regs[cpu_reg & 7].l = mask;
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVMSKPS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    ILLEGAL_ON(cpu_mod != 3);
    XMM_REG *s = &sse_xmm[cpu_rm & 7];
    uint32_t mask = 0;
    if (s->lo & 0x80000000ULL)        mask |= 1;
    if (s->lo & 0x8000000000000000ULL) mask |= 2;
    if (s->hi & 0x80000000ULL)        mask |= 4;
    if (s->hi & 0x8000000000000000ULL) mask |= 8;
    cpu_state.regs[cpu_reg & 7].l = mask;
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * MOVNTPS (0F 2B) – non-temporal packed store (treated like regular store)
 * ----------------------------------------------------------------------- */
static int
opMOVNTPS_a16(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_16(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememq(easeg, cpu_state.eaaddr,     sse_xmm[cpu_reg & 7].lo);
    writememq(easeg, cpu_state.eaaddr + 8, sse_xmm[cpu_reg & 7].hi);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(1);
    return 0;
}
static int
opMOVNTPS_a32(uint32_t fetchdat)
{
    SSE_ENTER();
    fetch_ea_32(fetchdat);
    ILLEGAL_ON(cpu_mod == 3);
    SEG_CHECK_WRITE(cpu_state.ea_seg);
    writememq(easeg, cpu_state.eaaddr,     sse_xmm[cpu_reg & 7].lo);
    writememq(easeg, cpu_state.eaaddr + 8, sse_xmm[cpu_reg & 7].hi);
    if (cpu_state.abrt) return 1;
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * PREFETCHT0/T1/T2/NTA  (0F 18)  –  all treated as NOP
 * ----------------------------------------------------------------------- */
static int
opPREFETCHSSE_a16(uint32_t fetchdat)
{
    fetch_ea_16(fetchdat);
    CLOCK_CYCLES(1);
    return 0;
}
static int
opPREFETCHSSE_a32(uint32_t fetchdat)
{
    fetch_ea_32(fetchdat);
    CLOCK_CYCLES(1);
    return 0;
}

/* -----------------------------------------------------------------------
 * Dispatcher for 0F prefix in REPE (F3) context → SSE scalar float ops
 * Called by opF3_0F_a16/a32 below, which are inserted in REPE tables.
 * ----------------------------------------------------------------------- */
static int
opF3_0F_a16(uint32_t fetchdat)
{
    uint32_t new_fetchdat = fastreadl(cs + cpu_state.pc);
    if (cpu_state.abrt) return 1;
    cpu_state.pc++;

    switch (new_fetchdat & 0xff) {
        case 0x10: return opMOVSS_load_a16(new_fetchdat >> 8);
        case 0x11: return opMOVSS_store_a16(new_fetchdat >> 8);
        case 0x2A: return opCVTSI2SS_a16(new_fetchdat >> 8);
        case 0x2C: return opCVTTSS2SI_a16(new_fetchdat >> 8);
        case 0x2D: return opCVTSS2SI_a16(new_fetchdat >> 8);
        case 0x51: return opSQRTSS_a16(new_fetchdat >> 8);
        case 0x52: return opRSQRTSS_a16(new_fetchdat >> 8);
        case 0x53: return opRCPSS_a16(new_fetchdat >> 8);
        case 0x58: return opADDSS_a16(new_fetchdat >> 8);
        case 0x59: return opMULSS_a16(new_fetchdat >> 8);
        case 0x5C: return opSUBSS_a16(new_fetchdat >> 8);
        case 0x5D: return opMINSS_a16(new_fetchdat >> 8);
        case 0x5E: return opDIVSS_a16(new_fetchdat >> 8);
        case 0x5F: return opMAXSS_a16(new_fetchdat >> 8);
        case 0xC2: return opCMPSS_a16(new_fetchdat >> 8);
        default:
            cpu_state.pc--;  /* back up over the second byte */
            /* Fall through to normal 0F handler */
            return x86_opcodes_0f[(0x0F) | cpu_state.op32](fetchdat);
    }
}

static int
opF3_0F_a32(uint32_t fetchdat)
{
    uint32_t new_fetchdat = fastreadl(cs + cpu_state.pc);
    if (cpu_state.abrt) return 1;
    cpu_state.pc++;

    switch (new_fetchdat & 0xff) {
        case 0x10: return opMOVSS_load_a32(new_fetchdat >> 8);
        case 0x11: return opMOVSS_store_a32(new_fetchdat >> 8);
        case 0x2A: return opCVTSI2SS_a32(new_fetchdat >> 8);
        case 0x2C: return opCVTTSS2SI_a32(new_fetchdat >> 8);
        case 0x2D: return opCVTSS2SI_a32(new_fetchdat >> 8);
        case 0x51: return opSQRTSS_a32(new_fetchdat >> 8);
        case 0x52: return opRSQRTSS_a32(new_fetchdat >> 8);
        case 0x53: return opRCPSS_a32(new_fetchdat >> 8);
        case 0x58: return opADDSS_a32(new_fetchdat >> 8);
        case 0x59: return opMULSS_a32(new_fetchdat >> 8);
        case 0x5C: return opSUBSS_a32(new_fetchdat >> 8);
        case 0x5D: return opMINSS_a32(new_fetchdat >> 8);
        case 0x5E: return opDIVSS_a32(new_fetchdat >> 8);
        case 0x5F: return opMAXSS_a32(new_fetchdat >> 8);
        case 0xC2: return opCMPSS_a32(new_fetchdat >> 8);
        default:
            cpu_state.pc--;
            return x86_opcodes_0f[(0x0F) | cpu_state.op32](fetchdat);
    }
}

#endif /* X86_OPS_SSE_H */
