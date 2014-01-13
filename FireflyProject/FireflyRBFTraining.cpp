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
    
    for (int i = 0; i < fireflyCount; i++) {
        std::vector<std::vector<double> > weights;
        for (int j = 0; j < rbfCount; j++) {
            std::vector<double> newVector(dim);
            for (auto &value : newVector) {
                value = score(mt);
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
        
        //            std::copy(fireflies.begin(), fireflies.end(), std::back_inserter(tmpFireflies));
        std::copy(fireflies.begin(), fireflies.end(), tmpFireflies.begin());
        
        for (auto &firefly : fireflies) {
            for (auto &tmpFirefly : tmpFireflies) {
                if (firefly.fitness < tmpFirefly.fitness) {
                    moveFirefly(firefly, tmpFirefly);
                }
            }
        }
        
        for (auto &firefly : fireflies) {
            firefly.calcFitness(input, output);
        }
        std::sort(fireflies.begin(), fireflies.end(), compare);
        
        Firefly &bestFirefly = fireflies[0];
        randomlyWalk(bestFirefly);
        bestFirefly.calcFitness(input, output);
        std::sort(fireflies.begin(), fireflies.end(), compare);
        
        gettimeofday(&t, NULL);
        std::cout << "iter = " << iter << ", maxFitness = " << bestFirefly.fitness
        << " [" << (t.tv_sec - s.tv_sec) * 1000 + (t.tv_usec - s.tv_usec) / 1000 << "ms]"
        << std::endl;
        
        ++iter;
    }
}

std::vector<double> FireflyRBFTraining::output(const std::vector<double> &input) {
    return fireflies[0].output(input);
}

void FireflyRBFTraining::outputBestFirefly(std::ofstream &output) {
//    Firefly bestFirefly = fireflies[0];
//
//    output << "double weights[][" << dim << "] = {";
//    auto wIter = bestFirefly.weights.begin();
//    while (wIter != bestFirefly.weights.end()) {
//        output << "{";
//        for (int i = 0; i < (*wIter).size(); i++) {
//            output << (*wIter)[i];
//            if (i < (*wIter).size() - 1) {
//                output << ", ";
//            }
//        }
//        output << "}";
//        ++wIter;
//        if (wIter != bestFirefly.weights.end()) {
//            output << ", ";
//        }
//    }
//    output << "};" << std::endl;
//    
//    output << "double centerVector[][" << dim << "] = {";
//    auto cIter = bestFirefly.centerVectors.begin();
//    while (cIter != bestFirefly.centerVectors.end()) {
//        output << "{";
//        for (int i = 0; i < (*cIter).size(); i++) {
//            output << (*cIter)[i];
//            if (i < (*cIter).size() - 1) {
//                output << ", ";
//            }
//        }
//        output << "}";
//        ++cIter;
//        if (cIter != bestFirefly.centerVectors.end()) {
//            output << ", ";
//        }
//    }
//    output << "};" << std::endl;
//    
//    output << "double spreads[] = {";
//    auto sIter = bestFirefly.spreads.begin();
//    while (sIter != bestFirefly.spreads.end()) {
//        output << *sIter;
//        ++sIter;
//        if (sIter != bestFirefly.spreads.end()) {
//            output << ", ";
//        }
//    }
//    output << "};" << std::endl;
//    
//    output << "double biases[] = {";
//    auto bIter = bestFirefly.biases.begin();
//    while (bIter != bestFirefly.biases.end()) {
//        output << *bIter;
//        ++bIter;
//        if (bIter != bestFirefly.biases.end()) {
//            output << ", ";
//        }
//    }
//    output << "};" << std::endl;
}


double FireflyRBFTraining::distanceBetweenTwoFireflies(const Firefly &firefly1, const Firefly &firefly2) {
    double radius = 0.0;
    
    di_w1IIter = firefly1.weights.begin();
    di_w1IIterEnd = firefly1.weights.end();
    di_w2IIter = firefly2.weights.begin();
    while (di_w1IIter != di_w1IIterEnd) {
        di_w1JIter = (*di_w1IIter).begin();
        di_w1JIterEnd = (*di_w1IIter).end();
        di_w2JIter = (*di_w2IIter).begin();
        while (di_w1JIter != di_w1JIterEnd) {
            radius = ((*di_w1JIter) - (*di_w2JIter)) * ((*di_w1JIter) - (*di_w2JIter));
            ++di_w1JIter;
            ++di_w2JIter;
        }
        ++di_w1IIter;
        ++di_w2IIter;
    }
    
    di_c1IIter = firefly1.centerVectors.begin();
    di_c1IIterEnd = firefly1.centerVectors.end();
    di_c2IIter = firefly2.centerVectors.begin();
    while (di_c1IIter != di_c1IIterEnd) {
        di_c1JIter = (*di_c1IIter).begin();
        di_c1JIterEnd = (*di_c1IIter).end();
        di_c2JIter = (*di_c2IIter).begin();
        while (di_c1JIter != di_c1JIterEnd) {
            radius = ((*di_c1JIter) - (*di_c2JIter)) * ((*di_c1JIter) - (*di_c2JIter));
            ++di_c1JIter;
            ++di_c2JIter;
        }
        ++di_c1IIter;
        ++di_c2IIter;
    }
    
    di_s1Iter = firefly1.spreads.begin();
    di_s1IterEnd = firefly1.spreads.end();
    di_s2Iter = firefly2.spreads.begin();
    while (di_s1Iter != di_s1IterEnd) {
        radius += ((*di_s1Iter) - (*di_s2Iter)) * ((*di_s1Iter) - (*di_s2Iter));
        ++di_s1Iter;
        ++di_s2Iter;
    }
    
    di_b1Iter = firefly1.biases.begin();
    di_b1IterEnd = firefly1.biases.end();
    di_b2Iter = firefly2.biases.begin();
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
    
    mo_wIIter = firefly.weights.begin();
    mo_wIIterEnd = firefly.weights.end();
    mo_wDIIter = destinationFirefly.weights.begin();
    while (mo_wIIter != mo_wIIterEnd) {
        mo_wJIter = (*mo_wIIter).begin();
        mo_wJIterEnd = (*mo_wIIter).end();
        mo_wDJIter = (*mo_wDIIter).begin();
        while (mo_wJIter != mo_wJIterEnd) {
            (*mo_wJIter) = (1.0 - beta) * (*mo_wJIter) + beta * (*mo_wDJIter) + alpha * (score(mt) - 0.5) * 2.0;
            ++mo_wJIter;
            ++mo_wDJIter;
        }
        ++mo_wIIter;
        ++mo_wDIIter;
    }
    
    mo_cIIter = firefly.centerVectors.begin();
    mo_cIIterEnd = firefly.centerVectors.end();
    mo_cDIIter = destinationFirefly.centerVectors.begin();
    while (mo_cIIter != mo_cIIterEnd) {
        mo_cJIter = (*mo_cIIter).begin();
        mo_cJIterEnd = (*mo_cIIter).end();
        mo_cDJIter = (*mo_cDIIter).begin();
        while (mo_cJIter != mo_cJIterEnd) {
            (*mo_cJIter) = (1.0 - beta) * (*mo_cJIter) + beta * (*mo_cDJIter) + alpha * (score(mt) - 0.5) * 2.0;
            ++mo_cJIter;
            ++mo_cDJIter;
        }
        ++mo_cIIter;
        ++mo_cDIIter;
    }
    
    mo_sIter = firefly.spreads.begin();
    mo_sIterEnd = firefly.spreads.end();
    mo_sDIter = destinationFirefly.spreads.begin();
    while (mo_sIter != mo_sIterEnd) {
        (*mo_sIter) = (1.0 - beta) * (*mo_sIter) + beta * (*mo_sDIter) + alpha * (score(mt) - 0.5) * 2.0;
        ++mo_sIter;
        ++mo_sDIter;
    }
    
    mo_bIter = firefly.biases.begin();
    mo_bIterEnd = firefly.biases.end();
    mo_bDIter = destinationFirefly.biases.begin();
    while (mo_bIter != mo_bIterEnd) {
        *mo_bIter = (1.0 - beta) * *mo_bIter + beta * *mo_bDIter + alpha * (score(mt) - 0.5) * 2.0;
        ++mo_bIter;
        ++mo_bDIter;
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


