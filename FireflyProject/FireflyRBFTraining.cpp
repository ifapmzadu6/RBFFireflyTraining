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
    this->alpha = gumma;
}

void FireflyRBFTraining::makeFireflyWithRandom() {
    std::random_device random;
    std::mt19937 mt(random());
    std::uniform_real_distribution<double> score(0.0, 1.0);
    std::uniform_real_distribution<double> mscore(-1.0, 1.0);
    
    std::vector<double> newVector(dim);
    for (int i = 0; i < fireflyCount; i++) {
        std::vector<std::vector<double>> weights;
        for (int j = 0; j < rbfCount; j++) {
            for (auto &value : newVector) value = mscore(mt);
            weights.push_back(newVector);
        }
        std::vector<std::vector<double>> centerVector;
        for (int j = 0; j < rbfCount; j++) {
            for (auto &value : newVector) value = score(mt);
            centerVector.push_back(newVector);
        }
        std::vector<double> spreads(rbfCount);
        for (auto &value : spreads) value = score(mt);
        std::vector<double> biases(dim);
        for (auto &value : biases) value = mscore(mt);
        
        auto newPtr = std::shared_ptr<Firefly>(new Firefly(dim, dataCount, rbfCount, attractiveness, attractivenessMin, gumma, weights, spreads, centerVector, biases));
        firefliesPtr.push_back(newPtr);
    }
}

void FireflyRBFTraining::makeFireflyWithInput(const std::vector<std::vector<double>> &inputs) {
    std::random_device random;
    std::mt19937 mt(random());
    std::uniform_real_distribution<double> score(0.0, 1.0);
    std::uniform_real_distribution<double> mscore(-1.0, 1.0);
    std::uniform_int_distribution<int> sscore(0, dataCount - 1);
    std::normal_distribution<double> nmscore(-1.0, 1.0);
    
    std::vector<double> newVector(dim);
    for (int i = 0; i < fireflyCount; i++) {
        std::vector<std::vector<double>> weights;
        for (int j = 0; j < rbfCount; j++) {
            for (auto &value : newVector) value = mscore(mt);
            weights.push_back(newVector);
        }
        std::vector<std::vector<double>> centerVector;
        for (int j = 0; j < rbfCount; j++) {
            std::vector<double> input = inputs[sscore(mt)];
            for (int k = 0; k < dim; k++) newVector[k] = input[k] + nmscore(mt) / 10.0;
            centerVector.push_back(newVector);
        }
        std::vector<double> spreads(rbfCount);
        for (auto &value : spreads) value = score(mt);
        std::vector<double> biases(dim);
        for (auto &value : biases) value = score(mt);
        
        auto newPtr = std::shared_ptr<Firefly>(new Firefly(dim, dataCount, rbfCount, attractiveness, attractivenessMin, gumma, weights, spreads, centerVector, biases));
        firefliesPtr.push_back(newPtr);
    }
}

void FireflyRBFTraining::makeFireflyWithData(const std::vector<std::vector<double>> &weights, const std::vector<std::vector<double>> &centerVector, const std::vector<double> &spreads, const std::vector<double> &biases) {
    firefliesPtr.resize(0);
    auto newPtr = std::shared_ptr<Firefly>(new Firefly(dim, dataCount, rbfCount, attractiveness, attractivenessMin, gumma, weights, spreads, centerVector, biases));
    firefliesPtr.push_back(newPtr);
}

void FireflyRBFTraining::training(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs) {
    std::cout << "-------Firefly Algorithm-------" << std::endl;
    std::cout << "Dimention = " << dim << " , RBFCount = " << rbfCount << " , FireflyCount = " << fireflyCount << " , MaxGeneration = " << maxGeneration << std::endl;
    std::cout << "[Start training!]" << std::endl;
    
    num_thread = std::thread::hardware_concurrency();
    
    std::random_device random;
    std::mt19937 mt(random());
    std::uniform_real_distribution<double> score(-1.0, 1.0);
    std::normal_distribution<double> nscore(-1.0, 1.0);
    std::uniform_real_distribution<double> eescore(0.0, 1.0);
    
    auto compare = [](std::shared_ptr<Firefly> const &obj1, std::shared_ptr<Firefly> const &obj2) {
        return obj1.get()->fitness > obj2.get()->fitness;
    };
    
    auto moveFireflyAsync = [&random, inputs, outputs](int begin, int end, std::vector<std::shared_ptr<Firefly>> firefliesPtr, std::vector<std::shared_ptr<Firefly>> tmpFirefliesPtr, double alpha) {
        std::mt19937 asymt(random());
        std::uniform_real_distribution<double> asyeescore(0.0, 1.0);
        std::uniform_real_distribution<double> asyscore(-1.0, 1.0);
        auto iter = firefliesPtr.begin();
        iter += begin;
        auto endIter = firefliesPtr.begin();
        endIter += end;
        while (iter != endIter) {
            auto ptr = (*iter).get();
            for (auto &tmpFireflyPtr : tmpFirefliesPtr) {
                auto tmpPtr = tmpFireflyPtr.get();
                if (ptr->fitness != tmpPtr->fitness) {
                    ptr->moveToFirefly(*tmpPtr, alpha, asymt, asyscore);
                }
                else {
                    break;
                }
            }
            ptr->findLimits();
            ptr->calcFitness(inputs, outputs, asymt, asyeescore);
            ++iter;
        }
    };
    
    //fitnessの計算
    for (auto &firefly : firefliesPtr) firefly.get()->calcFitness(inputs, outputs, mt, eescore);
    std::sort(firefliesPtr.begin(), firefliesPtr.end(), compare);
    
    int iter = 0;
    double delta = pow((pow(10.0, -4.0) / 0.9), 1.0 / (double)maxGeneration);
    std::vector<std::shared_ptr<Firefly>> tmpFirefliesPtr(fireflyCount);
    std::vector<std::future<void>> task(num_thread);
    
    while (iter < maxGeneration) {
        auto begin_time = std::chrono::high_resolution_clock::now();
        
        alpha = delta * alpha;
        
        //Fireflyをコピーする。後でFireflyの移動に使う。
        auto it = firefliesPtr.begin();
        auto itE = firefliesPtr.end();
        auto itT = tmpFirefliesPtr.begin();
        while (it != itE) {
            (*itT) = std::shared_ptr<Firefly>(new Firefly(*((*it).get())));
            ++it; ++itT;
        }
        
//        //最適なFireflyをランダムに移動させる
//        Firefly *bestFirefly = firefliesPtr[0].get();
//        bestFirefly->randomlyWalk(alpha, mt, score);
//        bestFirefly->findLimits();
//        bestFirefly->calcFitness(inputs, outputs, mt, eescore);
        
//        moveFireflyAsync(1, fireflyCount, firefliesPtr, tmpFirefliesPtr, alpha);
        
        int begin = 0;
        int count = fireflyCount;
        int a = (int)num_thread - 1;
        int b = (int)num_thread;
        double c = 1.0;
        int end = fireflyCount;
        for (int i = 0; i < num_thread; i++) {
            count = (c - c * sqrt(a) / sqrt(b)) * fireflyCount;
            begin = (i < num_thread - 1) ? end - count : 1;
//            std::cout << "begin = " << begin << "   end" << end << "   count" << count << std::endl;
            task[i] = std::async(moveFireflyAsync, begin, end, firefliesPtr, tmpFirefliesPtr, alpha);
            end = begin;
            c = c * sqrt(a) / sqrt(b);
            --a; --b;
//            std::cout << "a = " << a << "    b = " << b << std::endl;
        }
        for (auto &f : task) f.wait();
        
        //ソート
        std::sort(firefliesPtr.begin(), firefliesPtr.end(), compare);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "iter = " << iter << ", bestfitness = " << firefliesPtr[0].get()->fitness
        << "  [" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << "ms]"
        << std::endl;
        
        ++iter;
    }
}

std::vector<double> FireflyRBFTraining::output(const std::vector<double> &input) {
    Firefly *bestFirefly = firefliesPtr[0].get();
    return bestFirefly->output(input);
}

void FireflyRBFTraining::outputBestFirefly(std::ofstream &ofstream) {
    Firefly *bestFirefly = firefliesPtr[0].get();
    
    ofstream << "double weights[][" << dim << "] = {";
    auto wIter = bestFirefly->weights.begin();
    while (wIter != bestFirefly->weights.end()) {
        ofstream << "{";
        for (int i = 0; i < (*wIter).size(); i++) {
            ofstream << (*wIter)[i];
            if (i < (*wIter).size() - 1) {
                ofstream << ", ";
            }
        }
        ofstream << "}";
        ++wIter;
        if (wIter != bestFirefly->weights.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
    
    ofstream << "double centerVector[][" << dim << "] = {";
    auto cIter = bestFirefly->centerVectors.begin();
    while (cIter != bestFirefly->centerVectors.end()) {
        ofstream << "{";
        for (int i = 0; i < (*cIter).size(); i++) {
            ofstream << (*cIter)[i];
            if (i < (*cIter).size() - 1) {
                ofstream << ", ";
            }
        }
        ofstream << "}";
        ++cIter;
        if (cIter != bestFirefly->centerVectors.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
    
    ofstream << "double spreads[] = {";
    auto sIter = bestFirefly->spreads.begin();
    while (sIter != bestFirefly->spreads.end()) {
        ofstream << *sIter;
        ++sIter;
        if (sIter != bestFirefly->spreads.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
    
    ofstream << "double biases[] = {";
    auto bIter = bestFirefly->biases.begin();
    while (bIter != bestFirefly->biases.end()) {
        ofstream << *bIter;
        ++bIter;
        if (bIter != bestFirefly->biases.end()) {
            ofstream << ", ";
        }
    }
    ofstream << "};" << std::endl;
}
