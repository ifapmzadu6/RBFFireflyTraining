//
//  FireflyRBFTraining.cpp
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2014/01/13.
//  Copyright (c) 2014年 Keisuke Karijuku. All rights reserved.
//

#include "FireflyRBFTraining.h"

#include <iostream>
#include <chrono>

#include <thread>
#include <future>


FireflyRBFTraining::FireflyRBFTraining(int dim, int dataCount, int rbfCount, int fireflyCount, double attractiveness, double gumma, int maxGeneration)
: dim(dim), dataCount(dataCount), rbfCount(rbfCount), fireflyCount(fireflyCount), attractiveness(attractiveness), gumma(gumma), maxGeneration(maxGeneration) {
    this->attractivenessMin = 0.2;
    this->eps = 10e-6;
    this->alpha = gumma;
    std::random_device random;
    this->mt = std::mt19937(random());
    this->score = std::uniform_real_distribution<double>(0.0, 1.0);
    this->nscore = std::normal_distribution<double>(0.0, 1.0);
    
//    this->num_thread = std::thread::hardware_concurrency();
    this->num_thread = 2;
}

void FireflyRBFTraining::makeFireflyWithRandom() {
    fireflies = std::vector<Firefly>();
    std::uniform_real_distribution<double> mscore(-1.0, 1.0);
    
    std::vector<double> newVector(dim);
    for (int i = 0; i < fireflyCount; i++) {
        std::vector<std::vector<double>> weights;
        for (int j = 0; j < rbfCount; j++) {
            for (auto &value : newVector) {
                value = mscore(mt);
            }
            weights.push_back(newVector);
        }
        std::vector<std::vector<double>> centerVector;
        for (int j = 0; j < rbfCount; j++) {
            for (auto &value : newVector) {
                value = score(mt);
            }
            centerVector.push_back(newVector);
        }
        std::vector<double> spreads(rbfCount);
        for (auto &value : spreads) {
            value = score(mt);
        }
        std::vector<double> biases(dim);
        for (auto &value : biases) {
            value = score(mt);
        }
        Firefly newFirefly = Firefly(dim, dataCount, rbfCount, attractiveness, attractivenessMin, gumma, weights, spreads, centerVector, biases);
        fireflies.push_back(newFirefly);
    }
}

void FireflyRBFTraining::makeFireflyWithInput(std::vector<std::vector<double>> &inputs) {
    fireflies = std::vector<Firefly>();
    std::uniform_real_distribution<double> mscore(-1.0, 1.0);
    std::uniform_int_distribution<int> sscore(0, dataCount - 1);
    std::normal_distribution<double> nmscore(-1.0, 1.0);
    
    std::vector<double> newVector(dim);
    for (int i = 0; i < fireflyCount; i++) {
        std::vector<std::vector<double>> weights;
        for (int j = 0; j < rbfCount; j++) {
            for (auto &value : newVector) {
                value = mscore(mt);
            }
            weights.push_back(newVector);
        }
        std::vector<std::vector<double>> centerVector;
        for (int j = 0; j < rbfCount; j++) {
            std::vector<double> input = inputs[sscore(mt)];
            for (int k = 0; k < dim; k++) {
                newVector[k] = input[k] + nmscore(mt) / 10.0;
            }
            centerVector.push_back(newVector);
        }
        std::vector<double> spreads(rbfCount);
        for (auto &value : spreads) {
            value = score(mt);
        }
        std::vector<double> biases(dim);
        for (auto &value : biases) {
            value = score(mt);
        }
        Firefly newFirefly = Firefly(dim, dataCount, rbfCount, attractiveness, attractivenessMin, gumma, weights, spreads, centerVector, biases);
        fireflies.push_back(newFirefly);
    }
}

void FireflyRBFTraining::makeFireflyWithData(std::vector<std::vector<double>> &weights, std::vector<std::vector<double>> &centerVector, std::vector<double> &spreads, std::vector<double> &biases) {
    fireflies = std::vector<Firefly>();
    
    Firefly newFirefly = Firefly(dim, dataCount, rbfCount, attractiveness, attractivenessMin, gumma, weights, spreads, centerVector, biases);
    fireflies.push_back(newFirefly);
}

void FireflyRBFTraining::training(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs) {
    std::cout << "-------Firefly Algorithm-------" << std::endl;
    std::cout << "Dimention = " << dim << " , RBFCount = " << rbfCount << " , FireflyCount = " << fireflyCount << " , MaxGeneration = " << maxGeneration << std::endl;
    std::cout << "[Start training!]" << std::endl;
    
    //fitnessの計算
    for (auto &firefly : fireflies) {
        firefly.calcFitness(inputs, outputs);
    }
    std::sort(fireflies.begin(), fireflies.end(), Firefly::compare);
    
    int iter = 0;
    double delta = 1.0 - pow((pow(10.0, -4.0) / 0.9), 1.0 / (double)maxGeneration);
    std::vector<Firefly> tmpFireflies(fireflyCount);
    
    auto moveFireflyAsync = [&](int const &begin_index, int const &end_index) {
        auto iter = fireflies.begin();
        iter += begin_index;
        auto endIter = fireflies.begin();
        endIter += end_index;
        while (iter != endIter) {
            for (auto &tmpFirefly : tmpFireflies) {
                if ((*iter).fitness < tmpFirefly.fitness) {
                    (*iter).moveToFirefly(tmpFirefly, alpha, mt, score);
                }
            }
            (*iter).findLimits();
            (*iter).calcFitness(inputs, outputs);
            ++iter;
        }
    };
    
    while (iter < maxGeneration) {
        auto begin_time = std::chrono::high_resolution_clock::now();
        
        alpha = (1.0 - delta) * alpha;
        
        std::copy(fireflies.begin(), fireflies.end(), tmpFireflies.begin());
        
        std::vector<std::future<void>> task(num_thread);
        int num_block = fireflyCount / num_thread;
        int from = 0;
        int to = 0;
        for (int i = 0; i < num_thread; i++) {
            to = (i == num_thread - 1) ? fireflyCount - 1 : from + num_block - 1;
            task[i] = std::async(moveFireflyAsync, from, to);
            from = to + 1;
        }
        for (auto &f : task) {
            f.get();
        }
        
//        for (auto &firefly : fireflies) {
//            for (auto &tmpFirefly : tmpFireflies) {
//                if (firefly.fitness < tmpFirefly.fitness) {
//                    firefly.moveToFirefly(tmpFirefly, alpha, mt, score);
//                }
//            }
//            firefly.findLimits();
//            firefly.calcFitness(inputs, outputs);
//        }
        std::sort(fireflies.begin(), fireflies.end(), Firefly::compare);
        
        Firefly &bestFirefly = fireflies[0];
        bestFirefly.randomlyWalk(alpha, mt, nscore);
        bestFirefly.findLimits();    void randomlyWalk(double alpha, std::mt19937 &mt, std::uniform_real_distribution<double> &score);
        bestFirefly.calcFitness(inputs, outputs);
        std::sort(fireflies.begin(), fireflies.end(), Firefly::compare);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "iter = " << iter << ", bestfitness = " << fireflies[0].fitness
        << "  [" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << "ms]"
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
