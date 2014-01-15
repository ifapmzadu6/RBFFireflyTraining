//
//  FireflyRBFTraining.cpp
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2014/01/13.
//  Copyright (c) 2014年 Keisuke Karijuku. All rights reserved.
//

#include "FireflyRBFTraining.h"

void FireflyRBFTraining::makeFireflyWithRandom() {
    fireflies = std::vector<Firefly>();
    std::uniform_real_distribution<double> mscore(-1.0, 1.0);
    
    for (int i = 0; i < fireflyCount; i++) {
        std::vector<std::vector<double> > weights;
        for (int j = 0; j < rbfCount; j++) {
            std::vector<double> newVector(dim);
            for (auto &value : newVector) {
                value = mscore(mt);
            }
            weights.push_back(newVector);
        }
        std::vector<double> spreads(rbfCount);
        for (auto &value : spreads) {
            value = score(mt);
        }
        std::vector<std::vector<double> > centerVector;
        for (int j = 0; j < rbfCount; j++) {
            std::vector<double> newVector(dim);
            for (auto &value : newVector) {
                value = score(mt);
            }
            centerVector.push_back(newVector);
        }
        std::vector<double> biases(dim);
        for (auto &value : biases) {
            value = score(mt);
        }
        Firefly newFirefly = Firefly(dim, dataCount, rbfCount, weights, spreads, centerVector, biases);
        fireflies.push_back(newFirefly);
    }
}

void FireflyRBFTraining::makeFireflyWithData(std::vector<std::vector<double> > &weights, std::vector<std::vector<double> > &centerVector, std::vector<double> &spreads, std::vector<double> &biases) {
    fireflies = std::vector<Firefly>();
    
    Firefly newFirefly = Firefly(dim, dataCount, rbfCount, weights, spreads, centerVector, biases);
    fireflies.push_back(newFirefly);
}

void FireflyRBFTraining::training(const std::vector<std::vector<double> > &input, const std::vector<std::vector<double> > &output) {
    std::cout << "-------Firefly Algorithm-------" << std::endl;
    std::cout << "Dimention = " << dim << " , RBFCount = " << rbfCount << " , FireflyCount = " << fireflyCount << " , MaxGeneration = " << maxGeneration << std::endl;
    std::cout << "[Start training!]" << std::endl;
    
    //fitnessの計算
    for (auto &firefly : fireflies) {
        firefly.calcFitness(input, output);
    }
    std::sort(fireflies.begin(), fireflies.end(), compare);
    
    int iter = 0;
    double delta = 1.0 - pow((pow(10.0, -4.0) / 0.9), 1.0 / maxGeneration);
    
    std::vector<Firefly> tmpFireflies(fireflyCount);
    
    while (iter < maxGeneration) {
        struct timeval s, t;
        gettimeofday(&s, NULL);
        
        alpha = (1.0 - delta) * alpha;
        
        std::copy(fireflies.begin(), fireflies.end(), tmpFireflies.begin());
        
        for (auto &firefly : fireflies) {
            for (auto &tmpFirefly : tmpFireflies) {
                if (firefly.fitness < tmpFirefly.fitness) {
                    moveFirefly(firefly, tmpFirefly);
                }
            }
        }
        
        findLimits();
        
        for (auto &firefly : fireflies) {
            firefly.calcFitness(input, output);
        }
        std::sort(fireflies.begin(), fireflies.end(), compare);
        
//        Firefly &bestFirefly = fireflies[0];
//        randomlyWalk(bestFirefly);
//        bestFirefly.calcFitness(input, output);
//        std::sort(fireflies.begin(), fireflies.end(), compare);
        
        gettimeofday(&t, NULL);
        std::cout << "iter = " << iter << ", bestfitness = " << fireflies[0].fitness
        << "  [" << (t.tv_sec - s.tv_sec) * 1000 + (t.tv_usec - s.tv_usec) / 1000 << "ms]"
        << std::endl;
        
        ++iter;
    }
}

std::vector<double> FireflyRBFTraining::output(const std::vector<double> &input) {
    return fireflies[0].output(input);
}

void FireflyRBFTraining::outputBestFirefly(std::ofstream &ofstream) {
    Firefly bestFirefly = fireflies[0];
    
    ofstream << "double weights[][" << dim << "] = {";
    auto wIter = bestFirefly.weights.begin();
    while (wIter != bestFirefly.weights.end()) {
        ofstream << "{";
        for (int i = 0; i < (*wIter).size(); i++) {
            ofstream << (*wIter)[i];
            if (i < (*wIter).size() - 1) {
                ofstream << ", ";
            }
        }
        ofstream << "}";
        ++wIter;
        if (wIter != bestFirefly.weights.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
    
    ofstream << "double centerVector[][" << dim << "] = {";
    auto cIter = bestFirefly.centerVectors.begin();
    while (cIter != bestFirefly.centerVectors.end()) {
        ofstream << "{";
        for (int i = 0; i < (*cIter).size(); i++) {
            ofstream << (*cIter)[i];
            if (i < (*cIter).size() - 1) {
                ofstream << ", ";
            }
        }
        ofstream << "}";
        ++cIter;
        if (cIter != bestFirefly.centerVectors.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
    
    ofstream << "double spreads[] = {";
    auto sIter = bestFirefly.spreads.begin();
    while (sIter != bestFirefly.spreads.end()) {
        ofstream << *sIter;
        ++sIter;
        if (sIter != bestFirefly.spreads.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
    
    ofstream << "double biases[] = {";
    auto bIter = bestFirefly.biases.begin();
    while (bIter != bestFirefly.biases.end()) {
        ofstream << *bIter;
        ++bIter;
        if (bIter != bestFirefly.biases.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
}

double FireflyRBFTraining::distanceBetweenTwoFireflies(const Firefly &firefly1, const Firefly &firefly2) {
    double radius = 0.0;
    
    auto di_w1IIter = firefly1.weights.begin();
    auto di_w1IIterEnd = firefly1.weights.end();
    auto di_w2IIter = firefly2.weights.begin();
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
    
    auto di_c1IIter = firefly1.centerVectors.begin();
    auto di_c1IIterEnd = firefly1.centerVectors.end();
    auto di_c2IIter = firefly2.centerVectors.begin();
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
    
    auto di_s1Iter = firefly1.spreads.begin();
    auto di_s1IterEnd = firefly1.spreads.end();
    auto di_s2Iter = firefly2.spreads.begin();
    while (di_s1Iter != di_s1IterEnd) {
        radius += ((*di_s1Iter) - (*di_s2Iter)) * ((*di_s1Iter) - (*di_s2Iter));
        ++di_s1Iter;
        ++di_s2Iter;
    }
    
    auto di_b1Iter = firefly1.biases.begin();
    auto di_b1IterEnd = firefly1.biases.end();
    auto di_b2Iter = firefly2.biases.begin();
    while (di_b1Iter != di_b1IterEnd) {
        radius += ((*di_b1Iter) - (*di_b2Iter)) * ((*di_b1Iter) - (*di_b2Iter));
        ++di_b1Iter;
        ++di_b2Iter;
    }
    
    return sqrt(radius);
}

void FireflyRBFTraining::moveFirefly(Firefly &firefly, const Firefly &destinationFirefly) {
    double rij = distanceBetweenTwoFireflies(firefly, destinationFirefly);
    double beta = (attractiveness - attractivenessMin) * exp(-gumma * pow(rij, 2.0)) + attractivenessMin;
    
    auto mo_wIIter = firefly.weights.begin();
    auto mo_wIIterEnd = firefly.weights.end();
    auto mo_wDIIter = destinationFirefly.weights.begin();
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
    
    auto mo_cIIter = firefly.centerVectors.begin();
    auto mo_cIIterEnd = firefly.centerVectors.end();
    auto mo_cDIIter = destinationFirefly.centerVectors.begin();
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
    
    auto mo_sIter = firefly.spreads.begin();
    auto mo_sIterEnd = firefly.spreads.end();
    auto mo_sDIter = destinationFirefly.spreads.begin();
    while (mo_sIter != mo_sIterEnd) {
        (*mo_sIter) = (1.0 - beta) * (*mo_sIter) + beta * (*mo_sDIter) + alpha * (score(mt) - 0.5) * 2.0;
        if (*mo_sIter < -1.0) *mo_sIter = -1.0;
        else if (*mo_sIter > 1.0) *mo_sIter = 1.0;
        ++mo_sIter;
        ++mo_sDIter;
    }
    
    auto mo_bIter = firefly.biases.begin();
    auto mo_bIterEnd = firefly.biases.end();
    auto mo_bDIter = destinationFirefly.biases.begin();
    while (mo_bIter != mo_bIterEnd) {
        *mo_bIter = (1.0 - beta) * *mo_bIter + beta * *mo_bDIter + alpha * (score(mt) - 0.5) * 2.0;
        if (*mo_bIter < -1.0) *mo_bIter = -1.0;
        else if (*mo_bIter > 1.0) *mo_bIter = 1.0;
        ++mo_bIter;
        ++mo_bDIter;
    }
}

void FireflyRBFTraining::findLimits() {
    for (auto &firefly : fireflies) {
        for (auto &tmp : firefly.weights) {
            for (auto &value : tmp) {
                if (value < -1.0) value = -1.0;
                else if (value > 1.0) value = 1.0;
            }
        }
        for (auto &tmp : firefly.centerVectors) {
            for (auto &value : tmp) {
                if (value < -1.0) value = -1.0;
                else if (value > 1.0) value = 1.0;
            }
        }
        for (auto &value : firefly.spreads) {
            if (value < -1.0) value = -1.0;
            else if (value > 1.0) value = 1.0;
        }
        for (auto &value : firefly.biases) {
            if (value < -1.0) value = -1.0;
            else if (value > 1.0) value = 1.0;
        }
    }
}

void FireflyRBFTraining::randomlyWalk(Firefly &firefly) {
    for (auto &tmp : firefly.weights) {
        for (auto &value : tmp) {
            value += alpha * (score(mt) - 0.5) * 2.0;
        }
    }
    for (auto &tmp : firefly.centerVectors) {
        for (auto &value : tmp) {
            value += alpha * (score(mt) - 0.5) * 2.0;
        }
    }
    for (auto &value : firefly.spreads) {
        value += alpha * (score(mt) - 0.5) * 2.0;
    }
    for (auto &value : firefly.biases) {
        value += alpha * (score(mt) - 0.5) * 2.0;
    }
}


