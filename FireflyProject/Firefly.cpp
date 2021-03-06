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

#include "RBF.h"

Firefly::Firefly () {
}

Firefly::Firefly(int dim, int rbfCount, double attractiveness, double attractivenessMin, double gumma, const std::vector<std::vector<double>> &weights, const std::vector<double> &spreads, const std::vector<std::vector<double>> &centerVectors, const std::vector<double> &biases)
: dim(dim), rbfCount(rbfCount), attractiveness(attractiveness), attractivenessMin(attractivenessMin), gumma(gumma), weights(weights), spreads(spreads), centerVectors(centerVectors), biases(biases) {
}

Firefly::Firefly(const Firefly &obj)
: dim(obj.dim), rbfCount(obj.rbfCount), fitness(obj.fitness), attractiveness(obj.attractiveness), attractivenessMin(obj.attractivenessMin), gumma(obj.gumma), weights(obj.weights), spreads(obj.spreads), centerVectors(obj.centerVectors), biases(obj.biases) {
}

void Firefly::calcFitness(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs, std::mt19937 &mt, std::uniform_real_distribution<double> &score) {
    std::vector<std::vector<double>> tmpRbfOutput(inputs.size(), std::vector<double>(rbfCount, 0.0));
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
            ++cr_rJIter; ++cr_cIter; ++cr_sIter;
        }
        ++cr_rIIter; ++cr_iIter;
    }
    
    std::vector<std::vector<double>> tmpOutput(outputs.size(), std::vector<double>(dim, 0.0));
    mult(tmpOutput, tmpRbfOutput, weights);
    
    auto cf_oIIter = tmpOutput.begin();
    auto cf_oIIterEnd = tmpOutput.end();
    while (cf_oIIter != cf_oIIterEnd) {
        auto cf_oJIter = (*cf_oIIter).begin();
        auto cf_oJIterEnd = (*cf_oIIter).end();
        auto cf_bIter = biases.begin();
        while (cf_oJIter != cf_oJIterEnd) {
            (*cf_oJIter) += (*cf_bIter);
            ++cf_oJIter; ++cf_bIter;
        }
        ++cf_oIIter;
    }
    
    fitness = 1.0 / (1.0 + mse(tmpOutput, outputs));    
}

std::vector<double> Firefly::output(const std::vector<double> &input) const {
    std::vector<double> tmpRBFOutput(rbfCount, 0.0);
    auto tIter = tmpRBFOutput.begin();
    auto tIterEnd = tmpRBFOutput.end();
    auto cIter = centerVectors.begin();
    auto sIter = spreads.begin();
    while (tIter != tIterEnd) {
        (*tIter) = function(*sIter, *cIter, input);
        ++tIter; ++cIter; ++sIter;
    }
    
    std::vector<double> output(dim, 0.0);
    auto tVIter = tmpRBFOutput.begin();
    auto tVIterEnd = tmpRBFOutput.end();
    auto wIIter = weights.begin();
    while (tVIter != tVIterEnd) {
        auto oIter = output.begin();
        auto oIterEnd = output.end();
        auto wJIter = (*wIIter).begin();
        while (oIter != oIterEnd) {
            (*oIter) += (*wJIter) * (*tVIter);
            ++oIter; ++wJIter;
        }
        ++wIIter; ++tVIter;
    }
    
    auto ttIter = output.begin();
    auto ttIterEnd = output.end();
    auto bIter = biases.begin();
    while (ttIter != ttIterEnd) {
        (*ttIter) += (*bIter);
        ++ttIter; ++bIter;
    }
    
    return output;
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
            double d = (*di_w1JIter) - (*di_w2JIter);
            radius = d * d;
            ++di_w1JIter; ++di_w2JIter;
        }
        ++di_w1IIter; ++di_w2IIter;
    }
    
    auto di_c1IIter = centerVectors.begin();
    auto di_c1IIterEnd = centerVectors.end();
    auto di_c2IIter = firefly.centerVectors.begin();
    while (di_c1IIter != di_c1IIterEnd) {
        auto di_c1JIter = (*di_c1IIter).begin();
        auto di_c1JIterEnd = (*di_c1IIter).end();
        auto di_c2JIter = (*di_c2IIter).begin();
        while (di_c1JIter != di_c1JIterEnd) {
            double d = (*di_c1JIter) - (*di_c2JIter);
            radius = d * d;
            ++di_c1JIter; ++di_c2JIter;
        }
        ++di_c1IIter; ++di_c2IIter;
    }
    
    auto di_s1Iter = spreads.begin();
    auto di_s1IterEnd = spreads.end();
    auto di_s2Iter = firefly.spreads.begin();
    while (di_s1Iter != di_s1IterEnd) {
        double d = (*di_s1Iter) - (*di_s2Iter);
        radius += d * d;
        ++di_s1Iter; ++di_s2Iter;
    }
    
    auto di_b1Iter = biases.begin();
    auto di_b1IterEnd = biases.end();
    auto di_b2Iter = firefly.biases.begin();
    while (di_b1Iter != di_b1IterEnd) {
        double d = (*di_b1Iter) - (*di_b2Iter);
        radius += d * d;
        ++di_b1Iter; ++di_b2Iter;
    }
    
    return sqrt(radius);
}

void Firefly::moveToFirefly(const Firefly &firefly, double &alpha, std::mt19937 &mt, std::uniform_real_distribution<double> &score) {
    double rij = normToFirefly(firefly);
    double beta = (attractiveness - attractivenessMin) * exp(-gumma * pow(rij, 2.0)) + attractivenessMin;
    double cbeta = 1.0 - beta;
    
    auto mo_wIIter = weights.begin();
    auto mo_wIIterEnd = weights.end();
    auto mo_wDIIter = firefly.weights.begin();
    while (mo_wIIter != mo_wIIterEnd) {
        auto mo_wJIter = (*mo_wIIter).begin();
        auto mo_wJIterEnd = (*mo_wIIter).end();
        auto mo_wDJIter = (*mo_wDIIter).begin();
        while (mo_wJIter != mo_wJIterEnd) {
            (*mo_wJIter) = cbeta * (*mo_wJIter) + beta * (*mo_wDJIter) + alpha * score(mt);
            ++mo_wJIter; ++mo_wDJIter;
        }
        ++mo_wIIter; ++mo_wDIIter;
    }
    
    auto mo_cIIter = centerVectors.begin();
    auto mo_cIIterEnd = centerVectors.end();
    auto mo_cDIIter = firefly.centerVectors.begin();
    while (mo_cIIter != mo_cIIterEnd) {
        auto mo_cJIter = (*mo_cIIter).begin();
        auto mo_cJIterEnd = (*mo_cIIter).end();
        auto mo_cDJIter = (*mo_cDIIter).begin();
        while (mo_cJIter != mo_cJIterEnd) {
            (*mo_cJIter) = cbeta * (*mo_cJIter) + beta * (*mo_cDJIter) + alpha * score(mt);
            ++mo_cJIter; ++mo_cDJIter;
        }
        ++mo_cIIter; ++mo_cDIIter;
    }
    
    auto mo_sIter = spreads.begin();
    auto mo_sIterEnd = spreads.end();
    auto mo_sDIter = firefly.spreads.begin();
    while (mo_sIter != mo_sIterEnd) {
        (*mo_sIter) = cbeta * (*mo_sIter) + beta * (*mo_sDIter) + alpha * score(mt);
        ++mo_sIter; ++mo_sDIter;
    }
    
    auto mo_bIter = biases.begin();
    auto mo_bIterEnd = biases.end();
    auto mo_bDIter = firefly.biases.begin();
    while (mo_bIter != mo_bIterEnd) {
        (*mo_bIter) = cbeta * (*mo_bIter) + beta * (*mo_bDIter) + alpha * score(mt);
        ++mo_bIter; ++mo_bDIter;
    }
    
    findLimits();
}

void Firefly::randomlyWalk(double &alpha, std::mt19937 &mt, std::uniform_real_distribution<double> &score) {
    for (auto &tmp : weights) for (auto &value : tmp) value += alpha * score(mt);
    for (auto &tmp : centerVectors) for (auto &value : tmp) value += alpha * score(mt);
    for (auto &value : spreads) value += alpha * score(mt);
    for (auto &value : biases) value += alpha * score(mt);
    
    findLimits();
}

void Firefly::findLimits() {
    double ud = 1.0;
    double ld = -1.0;
    double zero = 0.0f;
    
    for (auto &tmp : weights) {
        for (auto &value : tmp) {
            if (value > ud) value = ud;
            else if (value < ld) value = ld;
        }
    }
    for (auto &tmp : centerVectors) {
        for (auto &value : tmp) {
            if (value > ud) value = ud;
            else if (value < ld) value = ld;
        }
    }
    for (auto &value : spreads) {
        if (value > ud) value = ud;
        else if (value < zero) value = zero;
    }
    
    for (auto &value : biases) {
        if (value > ud) value = ud;
        else if (value < ld) value = ld;
    }
}
