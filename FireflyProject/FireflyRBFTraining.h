//
//  Firefly.h
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2013/10/23.
//  Copyright (c) 2013年 Keisuke Karijuku. All rights reserved.
//

#ifndef FireflyProject_Firefly_h
#define FireflyProject_Firefly_h

#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <limits>

#include "Eigen/Dense"

class FireflyRBFTraining
{
public:
    class Firefly
    {
    public:
        //教師信号input,outputの次元
        int dim;
        //教師信号input,outputのデータ数
        int dataCount;
        //RBFの数
        int rbfCount;
        //RBFの重み(rbfCount, dim)
        Eigen::MatrixXd weights;
        //RBFのシグマ
        std::vector<double> spreads;
        //RBFのセンターベクトル
        std::vector<Eigen::VectorXd> centerVector;
        //outputのバイアス
        std::vector<double> biases;
        
        //適応度
        double fitness;
        
        Firefly(int dim, int dataCount, int rbfCount, const Eigen::MatrixXd &weights, const std::vector<double> spreads, const std::vector<Eigen::VectorXd> &centerVector, const std::vector<double> biases)
        : dim(dim), dataCount(dataCount), rbfCount(rbfCount), weights(weights), spreads(spreads), centerVector(centerVector), biases(biases) {
            
        }
        
        double function(const Eigen::VectorXd &X, int rbfIndex) const {
            return exp(- spreads[rbfIndex] * (centerVector[rbfIndex] - X).squaredNorm());
        }
        
        Eigen::VectorXd output(const Eigen::VectorXd &input) const {
            Eigen::VectorXd rbf_out(rbfCount);
            for(int i = 0; i < rbfCount; i++) {
                rbf_out(i) = function(input, i);
            }
            
            Eigen::VectorXd output = weights.transpose() * rbf_out;
            for (int i = 0; i < dim; i++) {
                output(i) += biases[i];
            }
            
            return output;
        }
        
        //未修整
//        void study(const std::vector<Eigen::VectorXd> &Input, const std::vector<Eigen::VectorXd> &Output) {
//            size_t n = Input.size();
//            Eigen::MatrixXd rbf_out(n, rbfCount);
//            Eigen::MatrixXd output(dim, n);
//            for(int i = 0; i < n; i++)
//                for(int j = 0; j < dim; j++)
//                    output(j, i) = Output[i](j);
//            for(int i = 0; i < n; i++)
//                for(int j = 0; j < rbfCount; j++)
//                    rbf_out(i, j) = function(Input[i]) + this->bias;
//            //Phi[N, unit] * W[unit, dim] = Output[N, dim] を解けば良い Wが未知数
//            this->weights = rbf_out.fullPivLu().solve(output.transpose());
//            //std::cout << W << std::endl;
//        }
    };
    
private:
    //最大試行回数
    int maxGeneration;
    //最大誤差
    double eps;
    //教師信号input,outputの次元
    int dim;
    //教師信号input,outputのデータ数
    int dataCount;
    //RBFの数
    int rbfCount;
    
    //firefly
	int fireflyCount;
    std::vector<Firefly> fireflies;
    
    //fireflyが移動する時の移動係数
    double attractiveness;
    double attractivenessMin;
    
    //fireflyが移動する時のランダム要素の倍率係数gumma
    double gumma;
    
    //乱数の影響力を決める
    double alpha;
    
    //NS-FAのfireflyの関係図のフラグ(fireflyCount, fireflyCount),Network-Structured Firefly Algorithmで使用
    std::vector<std::vector<bool>> connection;
    
    std::random_device random;
    std::mt19937 mt;
    std::uniform_real_distribution<double> score;
    
public:
    FireflyRBFTraining(int dim, int dataCount, int rbfCount, int fireflyCount, double attractiveness, double gumma, int maxGeneration)
    : dim(dim), dataCount(dataCount), rbfCount(rbfCount), fireflyCount(fireflyCount), attractiveness(attractiveness), gumma(gumma), maxGeneration(maxGeneration) {
        this->attractivenessMin = 0.2;
        this->eps = 10e-6;
        this->alpha = gumma;
        this->mt = std::mt19937(random());
        this->score = std::uniform_real_distribution<double>(0.0, 1.0);
        
        for (int i = 0; i < fireflyCount; i++) {
            Eigen::MatrixXd weights = Eigen::MatrixXd::Random(rbfCount, dim);
            std::vector<double> spreads(rbfCount);
            for (int j = 0; j < rbfCount; j++) {
                spreads[j] = score(mt);
            }
            std::vector<Eigen::VectorXd> centerVector(rbfCount);
            for (int j = 0; j < rbfCount; j++) {
                Eigen::VectorXd tmpVector = Eigen::VectorXd::Random(dim);
                centerVector[j] = tmpVector;
            }
            std::vector<double> biases(dim);
            for (int j = 0; j < dim; j++) {
                biases[j] = score(mt);
            }
            Firefly newFirefly = Firefly(dim, dataCount, rbfCount, weights, spreads, centerVector, biases);
            fireflies.push_back(newFirefly);
        }
        
        for (int i = 0; i < fireflyCount; i++) {
            std::vector<bool> tmpVector(fireflyCount);
            for (int j = 0; j < fireflyCount; j++) {
                tmpVector[j] = false;
            }
            connection.push_back(tmpVector);
        }
    }
    
    //教師信号入力input、教師信号出力output
    void training(const std::vector<Eigen::VectorXd> &input, const std::vector<Eigen::VectorXd> &output) {
        std::cout << "-------Firefly Algorithm-------" << std::endl;
        std::cout << "Dimention = " << dim << " , RBFCount = " << rbfCount << " , FireflyCount = " << fireflyCount << " , MaxGeneration = " << maxGeneration << std::endl;
        std::cout << "[Start training!]" << std::endl;
        
//        double cp;
        //fitnessの計算
        std::vector<Eigen::VectorXd> tmpInput(dataCount);
        for (int i = 0; i < fireflyCount; i++) {
            for (int k = 0; k < dataCount; k++) tmpInput[k] = fireflies[i].output(input[k]);
            fireflies[i].fitness = fitness(tmpInput, output);
        }
        std::sort(fireflies.begin(), fireflies.end(), compare);
        
        int iter = 0;
        double delta = 1.0-pow((pow(10.0, -4.0)/0.9), 1.0/maxGeneration);
        
        while (iter < maxGeneration) {
            alpha = (1-delta)*alpha;
            
            std::vector<Firefly> tmpFireflies;
            std::copy(fireflies.begin(), fireflies.end(), std::back_inserter(tmpFireflies));
            
            auto iterI = fireflies.begin();
            while (iterI != fireflies.end()) {
                auto iterJ = tmpFireflies.begin();
                while (iterJ != tmpFireflies.end()) {
                    if ((*iterI).fitness < (*iterJ).fitness) {
                        moveFirefly(*iterI, *iterJ);
                    }
                    iterJ++;
                }
                
                auto tmpInputIter = tmpInput.begin();
                auto inputIter = input.begin();
                while (tmpInputIter != tmpInput.end()) {
                    *tmpInputIter = (*iterI).output(*inputIter);
                    tmpInputIter++;
                    inputIter++;
                }
                (*iterI).fitness = fitness(tmpInput, output);
                iterI++;
            }
            
            std::sort(fireflies.begin(), fireflies.end(), compare);
            
//            //一番いいfireflyはランダムに移動
            Firefly bestFirefly = fireflies[0];
            randomlyWalk(bestFirefly);
            auto tmpInputIter = tmpInput.begin();
            auto inputIter = input.begin();
            while (tmpInputIter != tmpInput.end()) {
                *tmpInputIter = bestFirefly.output(*inputIter);
                tmpInputIter++;
                inputIter++;
            }
            bestFirefly.fitness = fitness(tmpInput, output);
            std::sort(fireflies.begin(), fireflies.end(), compare);
            
            std::cout << "iter = " << iter << ", maxFitness = " << fireflies[0].fitness << std::endl;
            iter++;
        }
    }
    
    void trainingNS () {
        //            cp = (double)iter / maxGeneration;
        
        //            for (int i = 0; i < fireflyCount; i++) {
        //                for (int j = 0; j < fireflyCount; j++) {
        //                    //もしfitnessが良ければそのfireflyに近づく
        //                    if (tmpFireflies[i].fitness < tmpFireflies[j].fitness) {
        //                        //下のコメントアウトを外せばNetwork-Structured Firefly Algorithm
        ////                        if (connection[i][j]) {
        //                            //fireflyを近づける
        //                            moveFirefly(fireflies[i], tmpFireflies[j]);
        ////                        }
        ////                        else if (score(mt) <= cp) connection[i][j] = true;
        //                    }
        ////                    else if(i != iMax) if (score(mt) <= cp) connection[i][j] = false;
        //                }
        ////                fitnesses[i] = fitness(tmpInput, output);
        ////                std::cout << (double)i / fireflyCount * 100 << "%...done." << std::endl;
        //            }
    }
    
    static bool compare(const Firefly& obj1, const Firefly& obj2) {
        return obj1.fitness < obj2.fitness;
    }
    
    //Fireflyの出力d、教師信号の出力o
    double fitness(const std::vector<Eigen::VectorXd> &d, const std::vector<Eigen::VectorXd> &o) const {
        return 1.0 / (1.0 + mse(d, o));
    }
    
    //Fireflyの出力d、教師信号の出力o
    double mse(const std::vector<Eigen::VectorXd> &d, const std::vector<Eigen::VectorXd> &o) const {
        double mse = 0.0;
        std::vector<Eigen::VectorXd>::const_iterator dIter = d.begin();
        std::vector<Eigen::VectorXd>::const_iterator oIter = o.begin();
        while (dIter != d.end()) {
            for (int j = 0; j < dim; j++) {
                mse += ((*dIter)(j) - (*oIter)(j)) * ((*dIter)(j) - (*oIter)(j));
            }
            dIter++;
            oIter++;
        }
        mse /= dataCount;
        return mse;
    }
    
    //各要素の距離
    double distanceBetweenTwoFireflies(const Firefly &firefly1, const Firefly &firefly2) const {
        double radius = 0.0;
        double tmp;
        
        for (int i = 0; i < rbfCount; i++) {
            for (int j = 0; j < dim; j++) {
                tmp = firefly1.weights(i, j) - firefly2.weights(i, j);
                radius += tmp * tmp;
            }
            
            for (int j = 0; j < dim; j++) {
                tmp = firefly1.centerVector[i](j) - firefly2.centerVector[i](j);
                radius += tmp * tmp;
            }
        }
        
        auto spreadsIter1 = firefly1.spreads.begin();
        auto spreadsIter2 = firefly2.spreads.begin();
        while (spreadsIter1 != firefly1.spreads.end()) {
            radius += (spreadsIter1 - spreadsIter2) * (spreadsIter1 - spreadsIter2);
            spreadsIter1++;
            spreadsIter2++;
        }
        
        auto biasesIter1 = firefly1.biases.begin();
        auto biasesIter2 = firefly2.biases.begin();
        while (biasesIter1 != firefly1.biases.end()) {
            radius += (biasesIter1 - biasesIter2) * (biasesIter1 - biasesIter2);
            biasesIter1++;
            biasesIter2++;
        }
        
        return sqrt(radius);
    }
    
    //firefly <- (1-beta)*firefly + beta*firefly + (rand-1/2)
    void moveFirefly(Firefly &firefly, const Firefly &destinationFirefly) {
        double rij = distanceBetweenTwoFireflies(firefly, destinationFirefly);
        double beta = (attractiveness - attractivenessMin) * exp(-gumma * pow(rij, 2.0)) + attractivenessMin;
        
        for (int i = 0; i < rbfCount; i++) {
            for (int j = 0; j < dim; j++) {
                firefly.weights(i, j) = (1 - beta) * firefly.weights(i, j) + beta * destinationFirefly.weights(i, j) + alpha * (score(mt) - 0.5) * 2;
            }
            
            firefly.spreads[i] = (1 - beta) * firefly.spreads[i] + beta * destinationFirefly.spreads[i] + alpha * (score(mt) - 0.5) * 2;
            
            for (int j = 0; j < dim; j++) {
                firefly.centerVector[i](j) = (1 - beta) * firefly.centerVector[i](j) + beta * destinationFirefly.centerVector[i](j) + alpha * (score(mt) - 0.5) * 2;
            }
        }
        
        for (int i = 0; i < dim; i++) {
            firefly.biases[i] = (1 - beta) * firefly.biases[i] + beta * destinationFirefly.biases[i] + alpha * (score(mt) - 0.5) * 2;
        }
    }
    
    //ランダムに移動　t <- t + sigma*(rand-1/2)
    void randomlyWalk(Firefly &firefly) {
        firefly.weights = Eigen::MatrixXd::Random(rbfCount, dim);
        
        auto spreadsIter = firefly.spreads.begin();
        while (spreadsIter != firefly.spreads.end()) {
            *spreadsIter = score(mt);
            spreadsIter++;
        }
        
        auto centerVectorIter = firefly.centerVector.begin();
        while (centerVectorIter != firefly.centerVector.end()) {
            *centerVectorIter = Eigen::VectorXd::Random(dim);
            centerVectorIter++;
        }
        
        auto biasesIter = firefly.biases.begin();
        while (biasesIter != firefly.biases.end()) {
            *biasesIter = score(mt);
            biasesIter++;
        }
        
//        for (int i = 0; i < rbfCount; i++) {
//            for (int j = 0; j < dim; j++) {
//                firefly.weights(i, j) += alpha * (score(mt) - 0.5) * 2;
//            }
//            
//            firefly.spreads[i] += alpha * (score(mt) - 0.5) * 2;
//            
//            for (int j = 0; j < dim; j++) {
//                firefly.centerVector[i](j) += alpha * (score(mt) - 0.5) * 2;
//            }
//        }
//        
//        for (int i = 0; i < dim; i++) {
//            firefly.biases[i] += alpha * (score(mt) - 0.5) * 2;
//        }
    }
    
    //一番いい結果のindexで出力
    Eigen::VectorXd output(const Eigen::VectorXd &input) const {
        return fireflies[0].output(input);
    }
    
    void outputBestFirefly(std::ofstream &output) {
        Firefly bestFirefly = fireflies[0];
        
        output << "double weights[][" << dim << "] = {";
        long rows = bestFirefly.weights.rows();
        long cols = bestFirefly.weights.cols();
        for (int i = 0; i < rows; i++) {
            output << "{";
            for (int j = 0; j < cols; j++) {
                output << bestFirefly.weights(i, j);
                if (j < cols - 1) {
                    output << ", ";
                }
            }
            if (i == rows - 1) {
                output << "}";
            }
            else {
                output << "}, ";
            }
        }
        output << "};" << std::endl;
        
        output << "double centerVector[][" << dim << "] = {";
        auto centerVectorIter = bestFirefly.centerVector.begin();
        while (centerVectorIter != bestFirefly.centerVector.end()) {
            output << "{";
            for (int i = 0; i < (*centerVectorIter).size(); i++) {
                output << (*centerVectorIter)[i];
                if (i < (*centerVectorIter).size() - 1) {
                    output << ", ";
                }
            }
            output << "}";
            centerVectorIter++;
            if (centerVectorIter != bestFirefly.centerVector.end()) {
                output << ", ";
            }
        }
        output << "};" << std::endl;
        
        output << "double spreads[] = {";
        auto spreadsIter = bestFirefly.spreads.begin();
        while (spreadsIter != bestFirefly.spreads.end()) {
            output << *spreadsIter;
            spreadsIter++;
            if (spreadsIter != bestFirefly.spreads.end()) {
                output << ", ";
            }
        }
        output << "};" << std::endl;
        
        output << "double biases[] = {";
        auto biasesIter = bestFirefly.biases.begin();
        while (biasesIter != bestFirefly.biases.end()) {
            output << *biasesIter;
            biasesIter++;
            if (biasesIter != bestFirefly.biases.end()) {
                output << ", ";
            }
        }
        output << "};" << std::endl;
    }
    
    void displayFirefly(int index) const {
        std::cout << "Firefly =" << std::endl;
        std::cout << "[weight]" << std::endl;
        std::cout << fireflies[index].weights << std::endl;
        std::cout << "[CenterVector]" << std::endl;
        for(auto centerVector : fireflies[index].centerVector)
            std::cout << centerVector[0] << std::endl;
        std::cout << "[spread]" << std::endl;
        for(auto spread : fireflies[index].spreads)
            std::cout << spread << std::endl;
        std::cout << "[bias]" << std::endl;
        for(auto bias : fireflies[index].biases)
            std::cout << bias << std::endl;
    }
};

#endif
