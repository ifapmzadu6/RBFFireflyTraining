//
//  Firefly.cpp
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2014/01/13.
//  Copyright (c) 2014年 Keisuke Karijuku. All rights reserved.
//

#include "Firefly.h"



Firefly::Firefly () {
    
}

Firefly::Firefly(int dim, int dataCount, int rbfCount, const std::vector<std::vector<double> > &weights, const std::vector<double> &spreads, const std::vector<std::vector<double> > &centerVectors, const std::vector<double> &biases) : dim(dim), dataCount(dataCount), rbfCount(rbfCount), weights(weights), spreads(spreads), centerVectors(centerVectors), biases(biases) {
    this->tmpRbfOutput = std::vector<std::vector<double> >();
    for (int i = 0; i < dataCount; i++) {
        std::vector<double> newVector = std::vector<double>(rbfCount, 0.0);
        this->tmpRbfOutput.push_back(newVector);
    }
    this->tmpOutput = std::vector<std::vector<double> >();
    for (int i = 0 ; i < dataCount; i++) {
        std::vector<double> newVector = std::vector<double>(dim, 0.0);
        this->tmpOutput.push_back(newVector);
    }
    
    this->tmpVector = std::vector<double>(rbfCount, 0.0);
    this->tmpVector1 = std::vector<double>(dim, 0.0);
}

inline double Firefly::norm(const std::vector<double> &a, const std::vector<double> &b) const{
    double d = 0.0;
    auto no_xIter = a.begin();
    auto no_xIterEnd = a.end();
    auto no_yIter = b.begin();
    while (no_xIter != no_xIterEnd) {
        d += ((*no_xIter) - (*no_yIter)) * ((*no_xIter) - (*no_yIter));
        ++no_xIter;
        ++no_yIter;
    }
    return d;
}

inline void Firefly::mult(std::vector<std::vector<double> > &Y, const std::vector<std::vector<double> > &A, const std::vector<std::vector<double> > &B) {
    for (auto &vec : Y) {
        for (auto &value : vec) {
            value = 0.0;
        }
    }
    
    auto mu_oIIter = Y.begin();
    auto mu_oIterEnd = Y.end();
    auto mu_rIIter = A.begin();
    while (mu_oIIter != mu_oIterEnd) {
        auto mu_rKIter = (*mu_rIIter).begin();
        auto mu_rKIterEnd = (*mu_rIIter).end();
        auto mu_wKIter = B.begin();
        while (mu_rKIter != mu_rKIterEnd) {
            auto mu_oJIter = (*mu_oIIter).begin();
            auto mu_oJIterEnd = (*mu_oIIter).end();
            auto mu_wJIter = (*mu_wKIter).begin();
            while (mu_oJIter != mu_oJIterEnd) {
                (*mu_oJIter) += (*mu_rKIter) * (*mu_wJIter);
                ++mu_oJIter;
                ++mu_wJIter;
            }
            ++mu_rKIter;
            ++mu_wKIter;
        }
        ++mu_oIIter;
        ++mu_rIIter;
    }
}

inline double Firefly::function(const double &spreads, const std::vector<double> &centerVector, const std::vector<double> &x) const {
    return exp(- spreads * norm(centerVector, x));
}

inline double Firefly::mse(const std::vector<std::vector<double> > &d, const std::vector<std::vector<double> > &o) const {
    double mse = 0.0;
    auto ms_dIIter = d.begin();
    auto ms_dIIterEnd = d.end();
    auto ms_oIIter = o.begin();
    while (ms_dIIter != ms_dIIterEnd) {
        auto ms_dJIter = (*ms_dIIter).begin();
        auto ms_dJIterEnd = (*ms_dIIter).end();
        auto ms_oJIter = (*ms_oIIter).begin();
        while (ms_dJIter != ms_dJIterEnd) {
            mse += (*ms_dJIter - *ms_oJIter) * (*ms_dJIter - *ms_oJIter);
            ++ms_dJIter;
            ++ms_oJIter;
        }
        ++ms_dIIter;
        ++ms_oIIter;
    }
    mse /= d.size();
    return mse;
}

inline void Firefly::calcRbfOutput(const std::vector<std::vector<double> > &inputs) {
    auto cr_rIIter = tmpRbfOutput.begin();
    auto cr_rIIterEnd = tmpRbfOutput.end();
    auto cr_iIter = inputs.begin();
    while (cr_rIIter != cr_rIIterEnd) {
        auto cr_rJIter = (*cr_rIIter).begin();
        auto cr_rJIterEnd = (*cr_rIIter).end();
        auto cr_cIter = centerVectors.begin();
        auto cr_sIter = spreads.begin();
        while (cr_rJIter != cr_rJIterEnd) {
            (*cr_rJIter) = function(*cr_sIter, *cr_cIter, *cr_iIter);
            ++cr_rJIter;
            ++cr_cIter;
            ++cr_sIter;
        }
        ++cr_rIIter;
        ++cr_iIter;
    }
}

void Firefly::calcFitness(const std::vector<std::vector<double> > &inputs, const std::vector<std::vector<double> > &outputs) {
    calcRbfOutput(inputs);
    mult(tmpOutput, tmpRbfOutput, weights);
    
    auto cf_oIIter = tmpOutput.begin();
    auto cf_oIIterEnd = tmpOutput.end();
    while (cf_oIIter != cf_oIIterEnd) {
        auto cf_oJIter = (*cf_oIIter).begin();
        auto cf_oJIterEnd = (*cf_oIIter).end();
        auto cf_bIter = biases.begin();
        while (cf_oJIter != cf_oJIterEnd) {
            (*cf_oJIter) += (*cf_bIter);
            ++cf_oJIter;
            ++cf_bIter;
        }
        ++cf_oIIter;
    }
    
    fitness = 1.0 / (1.0 + mse(tmpOutput, outputs));
}

std::vector<double> Firefly::output(const std::vector<double> &input) {
    auto tIter = tmpVector.begin();
    auto tIterEnd = tmpVector.end();
    auto cIter = centerVectors.begin();
    auto sIter = spreads.begin();
    while (tIter != tIterEnd) {
        (*tIter) = function(*sIter, *cIter, input);
        ++tIter;
        ++cIter;
        ++sIter;
    }
    
    for (auto &value : tmpVector1) {
        value = 0.0;
    }
    auto tVIter = tmpVector.begin();
    auto tVIterEnd = tmpVector.end();
    auto wIIter = weights.begin();
    while (tVIter != tVIterEnd) {
        auto oIter = tmpVector1.begin();
        auto oIterEnd = tmpVector1.end();
        auto wJIter = (*wIIter).begin();
        while (oIter != oIterEnd) {
            (*oIter) += (*wJIter) * (*tVIter);
            ++oIter;
            ++wJIter;
        }
        ++wIIter;
        ++tVIter;
    }
    
    auto ttIter = tmpVector1.begin();
    auto ttIterEnd = tmpVector1.end();
    auto bIter = biases.begin();
    while (ttIter != ttIterEnd) {
        (*ttIter) += (*bIter);
        ++ttIter;
        ++bIter;
    }
    
    return tmpVector1;
}


