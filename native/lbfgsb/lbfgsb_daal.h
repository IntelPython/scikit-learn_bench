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
         * iprint - Diagnostic information flag (see L-BFGS-B; 0 = silent).
         */
        Parameter(
            const dao::sum_of_functions::BatchPtr &function, // minimize this
            size_t nIterations = 100,
            double accuracyThreshold = 1.0e-05, // same as pgtol in L-BFGS-B
            dm::NumericTablePtr lowerBound = dm::NumericTablePtr(),
            dm::NumericTablePtr upperBound = dm::NumericTablePtr(),
            size_t batchSize = 1,
            int bounded = 0,
            double factr = 1e7,
            int iprint = 0
        ) :
            dai::Parameter(function, nIterations, accuracyThreshold, false, batchSize),
            lowerBound(lowerBound),
            upperBound(upperBound),
            bounded(bounded),
            factr(factr),
            iprint(iprint)
        {};


        ds::Status check() const {
            ds::Status s = dai::Parameter::check();
            if (!s) return s;

            DAAL_CHECK_EX(bounded < 0 || bounded > 4, ds::ErrorIncorrectParameter,
                          ds::ArgumentName, "bounded");
            if (bounded == 1 || bounded == 2) {
                // Require lower bound
            }
            if (bounded == 2 || bounded == 3) {
                // Require upper bound
            }

            DAAL_CHECK_EX(iprint >= 0, ds::ErrorIncorrectParameter,
                          ds::ArgumentName, "iprint");
            return s;
        }

        virtual ~Parameter() {};

        // Extra variables for L-BFGS-B.
        dm::NumericTablePtr lowerBound;
        dm::NumericTablePtr upperBound;
        int bounded;
        double factr;
        int iprint;

    };

    class BatchContainer : public da::AnalysisContainerIface<batch> {
        public:
            BatchContainer(ds::Environment::env *daalEnv) {
            }
            ~BatchContainer() {
            }

            ds::Status compute() {
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



