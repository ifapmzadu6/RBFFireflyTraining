//
//  Firefly.cpp
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2014/01/13.
//  Copyright (c) 2014年 Keisuke Karijuku. All rights reserved.
//

#include "Firefly.h"

#include <cmath>

#include <iostream>


Firefly::Firefly () {
}

Firefly::Firefly(int dim, int dataCount, int rbfCount, double attractiveness, double attractivenessMin, double gumma, const std::vector<std::vector<double>> &weights, const std::vector<double> &spreads, const std::vector<std::vector<double>> &centerVectors, const std::vector<double> &biases)
: dim(dim), dataCount(dataCount), rbfCount(rbfCount), attractiveness(attractiveness), attractivenessMin(attractivenessMin), gumma(gumma), weights(weights), spreads(spreads), centerVectors(centerVectors), biases(biases) {
}

Firefly::Firefly(const Firefly &obj)
: dim(obj.dim), dataCount(obj.dataCount), rbfCount(obj.rbfCount), fitness(obj.fitness), attractiveness(obj.attractiveness), attractivenessMin(obj.attractivenessMin), gumma(obj.gumma), weights(obj.weights), spreads(obj.spreads), centerVectors(obj.centerVectors), biases(obj.biases) {
}

void Firefly::calcFitness(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs) {
    std::vector<std::vector<double>> tmpRbfOutput(dataCount, std::vector<double>(rbfCount, 0.0));
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
    
    std::vector<std::vector<double>> tmpOutput(dataCount, std::vector<double>(dim, 0.0));
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

std::vector<double> Firefly::output(const std::vector<double> &input) const {
    std::vector<double> tmpVector(rbfCount, 0.0);
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
    
    std::vector<double> tmpVector1(dim, 0.0);
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

const double Firefly::function(const double &spreads, const std::vector<double> &centerVector, const std::vector<double> &x) const {
    return exp(- spreads * squeredNorm(centerVector, x));
}

const double Firefly::mse(const std::vector<std::vector<double>> &d, const std::vector<std::vector<double>> &o) const {
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

const double Firefly::squeredNorm(const std::vector<double> &a, const std::vector<double> &b) const {
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

void Firefly::mult(std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B) const {
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

const double Firefly::normToFirefly(const Firefly &firefly) const {
    double radius = 0.0;
    
    auto di_w1IIter = weights.begin();
    auto di_w1IIterEnd = weights.end();
    auto di_w2IIter = firefly.weights.begin();
    while (di_w1IIter != di_w1IIterEnd) {
        auto di_w1JIter = (*di_w1IIter).begin();
        auto di_w1JIterEnd = (*di_w1IIter).end();
        auto di_w2JIter = (*di_w2IIter).begin();
        while (di_w1JIter != di_w1JIterEnd) {
            radius = ((*di_w1JIter) - (*di_w2JIter)) * ((*di_w1JIter) - (*di_w2JIter));
            ++di_w1JIter;
            ++di_w2JIter;
        }
        ++di_w1IIter;
        ++di_w2IIter;
    }
    
    auto di_c1IIter = centerVectors.begin();
    auto di_c1IIterEnd = centerVectors.end();
    auto di_c2IIter = firefly.centerVectors.begin();
    while (di_c1IIter != di_c1IIterEnd) {
        auto di_c1JIter = (*di_c1IIter).begin();
        auto di_c1JIterEnd = (*di_c1IIter).end();
        auto di_c2JIter = (*di_c2IIter).begin();
        while (di_c1JIter != di_c1JIterEnd) {
            radius = ((*di_c1JIter) - (*di_c2JIter)) * ((*di_c1JIter) - (*di_c2JIter));
            ++di_c1JIter;
            ++di_c2JIter;
        }
        ++di_c1IIter;
        ++di_c2IIter;
    }
    
    auto di_s1Iter = spreads.begin();
    auto di_s1IterEnd = spreads.end();
    auto di_s2Iter = firefly.spreads.begin();
    while (di_s1Iter != di_s1IterEnd) {
        radius += ((*di_s1Iter) - (*di_s2Iter)) * ((*di_s1Iter) - (*di_s2Iter));
        ++di_s1Iter;
        ++di_s2Iter;
    }
    
    auto di_b1Iter = biases.begin();
    auto di_b1IterEnd = biases.end();
    auto di_b2Iter = firefly.biases.begin();
    while (di_b1Iter != di_b1IterEnd) {
        radius += ((*di_b1Iter) - (*di_b2Iter)) * ((*di_b1Iter) - (*di_b2Iter));
        ++di_b1Iter;
        ++di_b2Iter;
    }
    
    return sqrt(radius);
}

//template <typename Ta>
//void Firefly::moveToFirefly(const Firefly &firefly, double alpha, const Ta rand) {
void Firefly::moveToFirefly(const Firefly &firefly, double alpha, std::mt19937 &mt, std::uniform_real_distribution<double> &score) {
    double rij = normToFirefly(firefly);
    double beta = (attractiveness - attractivenessMin) * exp(-gumma * pow(rij, 2.0)) + attractivenessMin;
    
    auto mo_wIIter = weights.begin();
    auto mo_wIIterEnd = weights.end();
    auto mo_wDIIter = firefly.weights.begin();
    while (mo_wIIter != mo_wIIterEnd) {
        auto mo_wJIter = (*mo_wIIter).begin();
        auto mo_wJIterEnd = (*mo_wIIter).end();
        auto mo_wDJIter = (*mo_wDIIter).begin();
        while (mo_wJIter != mo_wJIterEnd) {
            (*mo_wJIter) = (1.0 - beta) * (*mo_wJIter) + beta * (*mo_wDJIter) + alpha * (score(mt) - 0.5) * 2.0;
            if (*mo_wJIter < -1.0) *mo_wJIter = -1.0;
            else if (*mo_wJIter > 1.0) *mo_wJIter = 1.0;
            ++mo_wJIter;
            ++mo_wDJIter;
        }
        ++mo_wIIter;
        ++mo_wDIIter;
    }
    
    auto mo_cIIter = centerVectors.begin();
    auto mo_cIIterEnd = centerVectors.end();
    auto mo_cDIIter = firefly.centerVectors.begin();
    while (mo_cIIter != mo_cIIterEnd) {
        auto mo_cJIter = (*mo_cIIter).begin();
        auto mo_cJIterEnd = (*mo_cIIter).end();
        auto mo_cDJIter = (*mo_cDIIter).begin();
        while (mo_cJIter != mo_cJIterEnd) {
            (*mo_cJIter) = (1.0 - beta) * (*mo_cJIter) + beta * (*mo_cDJIter) + alpha * (score(mt) - 0.5) * 2.0;
            if (*mo_cJIter < -1.0) *mo_cJIter = -1.0;
            else if (*mo_cJIter > 1.0) *mo_cJIter = 1.0;
            ++mo_cJIter;
            ++mo_cDJIter;
        }
        ++mo_cIIter;
        ++mo_cDIIter;
    }
    
    auto mo_sIter = spreads.begin();
    auto mo_sIterEnd = spreads.end();
    auto mo_sDIter = firefly.spreads.begin();
    while (mo_sIter != mo_sIterEnd) {
        (*mo_sIter) = (1.0 - beta) * (*mo_sIter) + beta * (*mo_sDIter) + alpha * (score(mt) - 0.5) * 2.0;
        if (*mo_sIter < -1.0) *mo_sIter = -1.0;
        else if (*mo_sIter > 1.0) *mo_sIter = 1.0;
        ++mo_sIter;
        ++mo_sDIter;
    }
    
    auto mo_bIter = biases.begin();
    auto mo_bIterEnd = biases.end();
    auto mo_bDIter = firefly.biases.begin();
    while (mo_bIter != mo_bIterEnd) {
        *mo_bIter = (1.0 - beta) * *mo_bIter + beta * *mo_bDIter + alpha * (score(mt) - 0.5) * 2.0;
        if (*mo_bIter < -1.0) *mo_bIter = -1.0;
        else if (*mo_bIter > 1.0) *mo_bIter = 1.0;
        ++mo_bIter;
        ++mo_bDIter;
    }
}

//template <typename Tb>
//void Firefly::randomlyWalk(double alpha, const Tb rand) {
void Firefly::randomlyWalk(double alpha, std::mt19937 &mt, std::normal_distribution<double> &score) {
    for (auto &tmp : weights) {
        for (auto &value : tmp) {
            value += alpha * (score(mt) - 0.5) * 2.0;
        }
    }
    for (auto &tmp : centerVectors) {
        for (auto &value : tmp) {
            value += alpha * (score(mt) - 0.5) * 2.0;
        }
    }
    for (auto &value : spreads) {
        value += alpha * (score(mt) - 0.5) * 2.0;
    }
    for (auto &value : biases) {
        value += alpha * (score(mt) - 0.5) * 2.0;
    }
}

void Firefly::findLimits() {
    for (auto &tmp : weights) {
        for (auto &value : tmp) {
            if (value < -1.0) value = -1.0;
            else if (value > 1.0) value = 1.0;
        }
    }
    for (auto &tmp : centerVectors) {
        for (auto &value : tmp) {
            if (value < -1.0) value = -1.0;
            else if (value > 1.0) value = 1.0;
        }
    }
    for (auto &value : spreads) {
        if (value < -1.0) value = -1.0;
        else if (value > 1.0) value = 1.0;
    }
    for (auto &value : biases) {
        if (value < -1.0) value = -1.0;
        else if (value > 1.0) value = 1.0;
    }
}

bool Firefly::compare(const Firefly &obj1, const Firefly &obj2) {
    return obj1.fitness < obj2.fitness;
};

