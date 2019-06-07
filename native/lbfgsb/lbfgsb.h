/*
 * Copyright (C) 2019 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

/*
 * lbfgsb.h
 *
 * C header for Fortran subroutine setulb provided by the L-BFGS-B library.
 */

#ifndef __LBFGSB_H__
#define __LBFGSB_H__

void setulb_(int *n, int *m, double *x, double *l, double *u, int *nbd,
            double *f, double *g, double *factr, double *pgtol, double *wa,
            int *iwa, char *task, int *iprint, char *csave, int *lsave,
            int *isave, double *dsave, long int task_len, long int csave_len);

#endif /* __LBFGSB_H__ */
