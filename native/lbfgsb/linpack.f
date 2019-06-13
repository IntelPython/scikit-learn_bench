c                                                                                      
c  L-BFGS-B is released under the “New BSD License” (aka “Modified BSD License”        
c  or “3-clause license”)                                                              
c  Please read attached file License.txt                                               
c                                        
c  These implementations have been replaced with calls to LAPACK as done
c  in scipy v.0.10.0. Scipy is also licensed under the BSD license:
c
c  Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
c  All rights reserved.
c  
c  Redistribution and use in source and binary forms, with or without
c  modification, are permitted provided that the following conditions
c  are met:
c  
c  1. Redistributions of source code must retain the above copyright
c     notice, this list of conditions and the following disclaimer.
c  
c  2. Redistributions in binary form must reproduce the above
c     copyright notice, this list of conditions and the following
c     disclaimer in the documentation and/or other materials provided
c     with the distribution.
c  
c  3. Neither the name of the copyright holder nor the names of its
c     contributors may be used to endorse or promote products derived
c     from this software without specific prior written permission.
c  
c  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
c  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
c  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
c  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
c  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
c  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
c  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
c  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
c  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
c  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
c  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

      subroutine dpofa(a,lda,n,info)
      integer lda,n,info
      double precision a(lda,1)
c
c     dpofa factors a double precision symmetric positive definite
c     matrix.
c
c     dpofa is usually called by dpoco, but it can be called
c     directly with a saving in time if  rcond  is not needed.
c     (time for dpoco) = (1 + 18/n)*(time for dpofa) .
c
c     on entry
c
c        a       double precision(lda, n)
c                the symmetric matrix to be factored.  only the
c                diagonal and upper triangle are used.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c     on return
c
c        a       an upper triangular matrix  r  so that  a = trans(r)*r
c                where  trans(r)  is the transpose.
c                the strict lower triangle is unaltered.
c                if  info .ne. 0 , the factorization is not complete.
c
c        info    integer
c                = 0  for normal return.
c                = k  signals an error condition.  the leading minor
c                     of order  k  is not positive definite.
c
c     This is a wrapper from scipy:
c     https://github.com/scipy/scipy/blob/
c    +37d1c4b8941469ed3ccec7ca362115488112b41b/scipy/optimize/lbfgsb/
c    +routines.f#L3991
      
      call dpotrf('U', n, a, lda, info)
      end
      
c====================== The end of dpofa ===============================

      subroutine dtrsl(t,ldt,n,b,job,info)
      integer ldt,n,job,info
      double precision t(ldt,1),b(1)
c
c
c     dtrsl solves systems of the form
c
c                   t * x = b
c     or
c                   trans(t) * x = b
c
c     where t is a triangular matrix of order n. here trans(t)
c     denotes the transpose of the matrix t.
c
c     on entry
c
c         t         double precision(ldt,n)
c                   t contains the matrix of the system. the zero
c                   elements of the matrix are not referenced, and
c                   the corresponding elements of the array can be
c                   used to store other information.
c
c         ldt       integer
c                   ldt is the leading dimension of the array t.
c
c         n         integer
c                   n is the order of the system.
c
c         b         double precision(n).
c                   b contains the right hand side of the system.
c
c         job       integer
c                   job specifies what kind of system is to be solved.
c                   if job is
c
c                        00   solve t*x=b, t lower triangular,
c                        01   solve t*x=b, t upper triangular,
c                        10   solve trans(t)*x=b, t lower triangular,
c                        11   solve trans(t)*x=b, t upper triangular.
c
c     on return
c
c         b         b contains the solution, if info .eq. 0.
c                   otherwise b is unaltered.
c
c         info      integer
c                   info contains zero if the system is nonsingular.
c                   otherwise info contains the index of
c                   the first zero diagonal element of t.
c
c     This is a wrapper from scipy:
c     https://github.com/scipy/scipy/blob/
c    +37d1c4b8941469ed3ccec7ca362115488112b41b/scipy/optimize/lbfgsb/
c    +routines.f#L4049
c

      character*1 uplo, trans
      
      if (job .eq. 00) then
          uplo = 'L'
          trans = 'N'
      else if (job .eq. 01) then
          uplo = 'U'
          trans = 'N'
      else if (job .eq. 10) then
          uplo = 'L'
          trans = 'T'
      else if (job .eq. 11) then
          uplo = 'U'
          trans = 'T'
      endif

      call dtrtrs(uplo, trans, 'N', n, 1, t, ldt, b, n, info)
      end
      
c====================== The end of dtrsl ===============================


