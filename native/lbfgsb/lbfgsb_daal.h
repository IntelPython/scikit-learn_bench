/*
 * Copyright (C) 2019 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

/*
 * lbfgsb_daal.h
 *
 * Iterative solver class for DAAL algorithms using the L-BFGS-B library.
 */

#include "daal.h"
#include "lbfgsb.h"

namespace da = daal::algorithms;
namespace dai = daal::algorithms::optimization_solver::iterative_solver;
namespace dao = daal::algorithms::optimization_solver;
namespace dm = daal::data_management;
namespace ds = daal::services;
using namespace daal;
using namespace daal::algorithms;

namespace lbfgsb {

    enum Method {
        defaultDense = 0
    };

    struct Parameter : public dai::Parameter {

        /*
         * function - function to minimize
         * nIterations - same as maxiter in scipy.optimize.fmin_l_bfgs_b
         * accuracyThreshold - same as tol (scipy) or pgtol (L-BFGS-B)
         * lowerBound - same as l (L-BFGS-B). Lower bounds on variables.
         * upperBound - same as u (L-BFGS-B). Upper bounds on variables.
         * batchSize - ignored
         * bounded - same as nbd (L-BFGS-B).
         *      bounded[i] = 0 if x[i] unbounded,
         *                   1         lower bounded,
         *                   2         lower and upper bounded,
         *                   3         only upper bounded.
         * factr - Tolerance in termination test (see L-BFGS-B).
         * m - Number of corrections used in limited memory matrix (see L-BFGS-B).
         * iprint - Diagnostic information flag (see L-BFGS-B; 0 = silent).
         */
        Parameter(
            const dao::sum_of_functions::BatchPtr &function, // minimize this
            size_t nIterations = 100,
            double accuracyThreshold = 1.0e-05, // same as pgtol in L-BFGS-B
            dm::NumericTablePtr lowerBound = dm::NumericTablePtr(),
            dm::NumericTablePtr upperBound = dm::NumericTablePtr(),
            size_t batchSize = 1,
            dm::NumericTablePtr bounded = dm::NumericTablePtr(),
            double factr = 1e7,
            int m = 10,
            int iprint = 0
        ) :
            dai::Parameter(function, nIterations, accuracyThreshold, false, batchSize),
            lowerBound(lowerBound),
            upperBound(upperBound),
            bounded(bounded),
            factr(factr),
            m(m),
            iprint(iprint)
        {};


        ds::Status check() const {
            ds::Status s = dai::Parameter::check();
            if (!s) return s;

            return s;
        }

        virtual ~Parameter() {};

        // Extra variables for L-BFGS-B.
        dm::NumericTablePtr lowerBound;
        dm::NumericTablePtr upperBound;
        dm::NumericTablePtr bounded;
        double factr;
        int m;
        int iprint;

    };

    class BatchContainer : public da::AnalysisContainerIface<batch> {
        public:
            BatchContainer(ds::Environment::env *daalEnv) {
            }
            ~BatchContainer() {
            }
            ds::Status compute() {

                // If following along, this is similar to the code
                // in driver1.f provided in the L-BFGS-B library.

                // Fetch parameters so we don't have to go through
                // hoops to get them.
                dai::Input *input = static_cast<dai::Input *>(_in);
                dai::Result *result = static_cast<dai::Result *>(_res);
                Parameter *parameter = static_cast<Parameter *>(_par);
                dao::sum_of_functions::BatchPtr function = parameter->function;

                // We need value and gradient for L-BFGS-B
                function->sumOfFunctionsParameter->resultsToCompute =
                    dao::objective_function::value | dao::objective_function::gradient;

                // The initial argument to the function to minimize.
                dm::NumericTablePtr inputArgument = input->get(dai::inputArgument);
                // We will place the computed minimizing argument here
                dm::NumericTablePtr minimum = result->get(dai::minimum);
                // We will write the number of iterations used here.
                dm::NumericTablePtr actualIters = result->get(dai::nIterations);

                // Read parameters.
                const double accuracyThreshold = parameter->accuracyThreshold;
                dm::NumericTablePtr lowerBound = parameter->lowerBound;
                dm::NumericTablePtr upperBound = parameter->upperBound;
                dm::NumericTablePtr bounded = parameter->bounded;
                double factr = parameter->factr;
                int iprint = parameter->iprint;
                size_t nIter = parameter->nIterations;


                // Static allocations which don't change.
                char task[60], csave[60];
                int lsave[4];
                int n, m, isave[44];
                double f, pgtol, dsave[29];

                n = inputArgument->getNumberOfRows();
                m = parameter->m;

                // Because we have the freedom to dynamically allocate
                // our arrays, we don't need to specify nmax, mmax as
                // in driver1.f.
                int *nbd = new int[n];
                int *iwa = new int[3*n];
                double *x = new double[n];
                double *l = new double[n];
                double *u = new double[n];
                double *g = new double[n];
                double *wa = new double[2*m*n + 5*n + 11*m*m + 8*m];

                // set bounds in nbd, l, u.
                dm::BlockDescriptor<double> block;
                double *blockPtr;
                if (bounded) {
                    bounded->getBlockOfRows(0, 1, dm::readOnly, block);
                    blockPtr = block.getBlockPtr();
                    memcpy(nbd, blockPtr, n*sizeof(double));
                    bounded->releaseBlockOfRows(block);

                    lowerBound->getBlockOfRows(0, 1, dm::readOnly, block);
                    blockPtr = block.getBlockPtr();
                    memcpy(l, blockPtr, n*sizeof(double));
                    lowerBound->releaseBlockOfRows(block);

                    upperBound->getBlockOfRows(0, 1, dm::readOnly, block);
                    blockPtr = block.getBlockPtr();
                    memcpy(u, blockPtr, n*sizeof(double));
                    upperBound->releaseBlockOfRows(block);
                } else {
                    for (int i = 0; i < n; i++) {
                        nbd[i] = 0.;
                    }
                }

                // set initial guess of x
                inputArgument->getBlockOfRows(0, n, dm::readOnly, block);
                blockPtr = block.getBlockPtr();
                memcpy(x, blockPtr, n*sizeof(double));
                inputArgument->releaseBlockOfRows(block);

                // set task
                strcpy(task, "START");
                memset(task+5, ' ', sizeof(task) - 5);

                // loop: setulb, compute function, compute gradient.
                size_t actual_n_iterations = 0;
                do {
                    // This is the actual function call to the L-BFGS-B library.
                    setulb_(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa,
                            iwa, task, &iprint, csave, lsave, isave, dsave,
                            60, 60);

                    if (strncmp(task, "FG", 2) == 0) {

                        // The library asked us to compute the function value
                        // and its gradient.
                        size_t rows = inputArgument->getNumberOfRows();
                        dm::NumericTablePtr x_nt = dm::HomogenNumericTable<double>::create(
                                x, 1, rows);

                        function->sumOfFunctionsInput->set(
                                dao::sum_of_functions::argument, x_nt);
                        function->computeNoThrow();

                        // Get the function value.
                        dm::NumericTablePtr f_nt = function->getResult()->get(
                                dao::objective_function::valueIdx);
                        f_nt->getBlockOfRows(0, 1, dm::readOnly, block);
                        blockPtr = block.getBlockPtr();
                        f = *blockPtr;
                        f_nt->releaseBlockOfRows(block);

                        // Get the gradient value.
                        dm::NumericTablePtr g_nt = function->getResult()->get(
                                dao::objective_function::gradientIdx);
                        g_nt->getBlockOfRows(0, n, dm::readOnly, block);
                        blockPtr = block.getBlockPtr();
                        memcpy(g, blockPtr, n*sizeof(double));
                        g_nt->releaseBlockOfRows(block);

                    } else if (strncmp(task, "NEW_X", 5) == 0) {

                        // New iteration.
                        actual_n_iterations++;
                        if (actual_n_iterations >= nIter) {
                            strcpy(task, "STOP ");
                        }
                    } else {
                        break;
                    }

                } while (1);

                minimum->getBlockOfRows(0, n, dm::readWrite, block);
                blockPtr = block.getBlockPtr();
                memcpy(blockPtr, x, n*sizeof(double));
                minimum->releaseBlockOfRows(block);

                actualIters->getBlockOfRows(0, 1, dm::readWrite, block);
                blockPtr = block.getBlockPtr();
                *blockPtr = actual_n_iterations;
                actualIters->releaseBlockOfRows(block);

                return ds::Status();

            }
    };

    class Batch : public dai::Batch {
        public:
            typedef lbfgsb::Parameter ParameterType;
            typedef dai::Input InputType;
            typedef dai::Result ResultType;

            InputType input;
            Parameter parameter;

            Batch(const dao::sum_of_functions::BatchPtr &func = dao::sum_of_functions::BatchPtr()) :
                input(),
                parameter(func)
            {
                initialize();
            }

            Batch(const Batch &other) :
                dai::Batch(other),
                input(other.input),
                parameter(other.parameter)
            {
                initialize();
            }

            int getMethod() const { return 0; }
            dai::Input *getInput() { return &input; }
            dai::Parameter *getParameter() { return &parameter; }

            ds::Status createResult() {
                _result = dai::ResultPtr(new ResultType());
                _res = NULL;
                return ds::Status();
            }

            ds::SharedPtr<Batch> clone() const {
                return ds::SharedPtr<Batch>(cloneImpl());
            }

            static ds::SharedPtr<Batch> create();

        protected:
            Batch *cloneImpl() const {
                return new Batch(*this);
            }

            ds::Status allocateResult() {
                ds::Status s = static_cast<ResultType *>(_result.get())
                    ->allocate<double>(&input, &parameter, 0);
                _res = _result.get();
                return s;
            }

            void initialize() {
                // this is of type AlgorithmContainerImpl<batch>
                // Because BatchContainer inherits from that eventually,
                // let's just use that instead of AlgorithmDispatchContainer.
                Analysis<batch>::_ac = new BatchContainer(&_env);
                _par = &parameter;
                _in = &input;
                _result = dai::ResultPtr(new ResultType());
            }

    };

}



