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
        
    public:
        Firefly(int dim, int dataCount, int rbfCount, const Eigen::MatrixXd &weights, const std::vector<double> spreads, const std::vector<Eigen::VectorXd> &centerVector, const std::vector<double> biases)
        {
            this->dim = dim;
            this->dataCount = dataCount;
            this->rbfCount = rbfCount;
            this->weights = weights;
            this->spreads = spreads;
            this->centerVector = centerVector;
            this->biases = biases;
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
    int iteration;
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
    double sigma;
    
    //NS-FAのfireflyの関係図のフラグ(fireflyCount, fireflyCount),Network-Structured Firefly Algorithmで使用
    std::vector<std::vector<bool>> connection;
    
    //一番結果のいいfireflyのindex
    int bestFireflyIndex;
    
    std::random_device random;
    
    
public:
    FireflyRBFTraining(int dim, int dataCount, int rbfCount, int fireflyCount, double attractiveness, double gumma, int iteration) {
        this->iteration = iteration;
        this->dim = dim;
        this->dataCount = dataCount;
        this->rbfCount = rbfCount;
        this->fireflyCount = fireflyCount;
        this->attractiveness = attractiveness;
        this->attractivenessMin = 0.2;
        this->gumma = gumma;
        this->bestFireflyIndex = 0;
        this->sigma = 0.5;
        this->eps = 10e-6;
        
        std::mt19937 mt(random());
        std::uniform_real_distribution<double> score(0.0, 1.0);
        
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
                tmpVector[j] = true;
            }
            connection.push_back(tmpVector);
        }
    }
    
    //教師信号入力input、教師信号出力output
    void training(const std::vector<Eigen::VectorXd> &input, const std::vector<Eigen::VectorXd> &output) {
        std::cout << "Start training!" << std::endl;
        
        std::mt19937 mt(random());
        std::uniform_real_distribution<double> score(0.0, 1.0);
        std::vector<Eigen::VectorXd> tmpInput(dataCount);
        
        int iMax = 0;
        double tmpf = 0.0;
        int iter = 0;
        double cp;
        double sigmaT = sigma * (1.0 - pow((10e-4 / 0.9), (1.0 / iteration)));
        double rij, beta;
        //fitnessの値のキャッシュ
        std::vector<double> fitnesses(fireflyCount);
        for (int i = 0; i < fireflyCount; i++) {
            fitnesses[i] = 0.0;
        }
        while (iter < iteration) {
            cp = (double)iter / (double)iteration;
            for (int i = 0; i < fireflyCount; i++) {
                for (int j = 0; j < fireflyCount; j++) {
                    //fireflyを選んでfitness計算。キャッシュになければ計算
                    if (fitnesses[i] == 0.0) {
                        for (int k = 0; k < dataCount; k++) {
                            tmpInput[k] = fireflies[i].output(input[k]);
                        }
                        fitnesses[i] = fitness(tmpInput, output);
                    }
                    if (fitnesses[j] == 0.0) {
                        for (int k = 0; k < dataCount; k++) {
                            tmpInput[k] = fireflies[j].output(input[k]);
                        }
                        fitnesses[j] = fitness(tmpInput, output);
                    }
                    
                    //もしfitnessが良ければそのfireflyに近づく
                    if (fitnesses[i] < fitnesses[j]) {
                        //下のコメントアウトを外せばNetwork-Structured Firefly Algorithm
//                        if (connection[i][j]) {
                            rij = distanceBetweenTwoFireflies(fireflies[i], fireflies[j]);
                            beta = (attractiveness - attractivenessMin) * exp(-gumma * rij) + attractivenessMin;
                            moveFirefly(fireflies[i], fireflies[j], beta, sigmaT, mt, score);
                            //移動することでfitnessの値が変わるのでキャッシュから削除
                            fitnesses[i] = 0.0;
//                        }
//                        else if (score(mt) <= cp)
//                            connection[i][j] = true;
                    }
//                    else if(i != iMax)
//                        if (score(mt) <= cp)
//                            connection[i][j] = false;
                }
//                std::cout << (double)i / fireflyCount * 100 << "%...done." << std::endl;
            }
            //一番いい結果のindexを検索
            tmpf = 0.0;
            iMax = 0;
            for (int i = 0; i < fireflyCount; i++) {
                if (fitnesses[i] == 0.0) {
                    for (int j = 0; j < dataCount; j++) {
                        tmpInput[j] = fireflies[i].output(input[j]);
                    }
                    fitnesses[i] = fitness(tmpInput, output);
                }
                if (tmpf < fitnesses[i]) {
                    tmpf = fitnesses[i];
                    iMax = i;
                }
            }
            //一番いいfireflyはランダムに移動
            randomlyWalk(fireflies[iMax], score, mt, sigmaT);
            //キャッシュから削除
            fitnesses[iMax] = 0.0;
            
            std::cout << "iter = " << iter << ", maxI = " << iMax << ", maxFitness = " << tmpf << std::endl;
//            this->displayFirefly(iMax);
            
            iter++;
        }
        //一番結果がいいindexを保存
        bestFireflyIndex = iMax;
    }
    
    //Fireflyの出力d、教師信号の出力o
    double fitness(const std::vector<Eigen::VectorXd> &d, const std::vector<Eigen::VectorXd> &o) const {
        return 1.0 / (1.0 + mse(d, o));
    }
    
    //Fireflyの出力d、教師信号の出力o
    double mse(const std::vector<Eigen::VectorXd> &d, const std::vector<Eigen::VectorXd> &o) const {
        double mse = 0.0;
        
        for (int i = 0; i < dataCount; i++) {
            for (int j = 0; j < dim; j++) {
                mse += (d[i](j) - o[i](j)) * (d[i](j) - o[i](j));
            }
        }
        
        mse /= dataCount;
        
        return mse;
    }
    
    //各要素の距離
    double distanceBetweenTwoFireflies(const Firefly &firefly1, const Firefly &firefly2) const {
        double radius = 0.0;
        
        for (int i = 0; i < rbfCount; i++) {
            for (int j = 0; j < dim; j++) {
                double tmp = firefly1.weights(i, j) - firefly2.weights(i, j);
                radius += tmp * tmp;
            }
            
            double tmp1 = (firefly1.spreads[i] - firefly2.spreads[i]);
            radius += tmp1 * tmp1;
            
            for (int j = 0; j < dim; j++) {
                double tmp = firefly1.centerVector[i](j) - firefly2.centerVector[i](j);
                radius += tmp * tmp;
            }
        }
        
        for (int i = 0; i < dim; i++) {
            double tmp2 = (firefly1.biases[i] - firefly2.biases[i]);
            radius += tmp2 * tmp2;
        }
        
        return radius;
    }
    
    //firefly <- (1-beta)*firefly + beta*firefly + (rand-1/2)
    void moveFirefly(Firefly &firefly, const Firefly &destinationFirefly, double beta, double sigma, std::mt19937 mt, std::uniform_real_distribution<double> score) {
        for (int i = 0; i < rbfCount; i++) {
            for (int j = 0; j < dim; j++) {
                firefly.weights(i, j) = (1 - beta) * firefly.weights(i, j) + beta * destinationFirefly.weights(i, j) + sigma * (score(mt) - 0.5);
            }
            
            firefly.spreads[i] = (1 - beta) * firefly.spreads[i] + beta * destinationFirefly.spreads[i] + sigma * (score(mt) - 0.5);
            
            for (int j = 0; j < dim; j++) {
                firefly.centerVector[i](j) = (1 - beta) * firefly.centerVector[i](j) + beta * destinationFirefly.centerVector[i](j) + sigma * (score(mt) - 0.5);
            }
        }
        
        for (int i = 0; i < dim; i++) {
            firefly.biases[i] = (1 - beta) * firefly.biases[i] + beta * destinationFirefly.biases[i] + sigma * (score(mt) - 0.5);
        }
    }
    
    //ランダムに移動　t <- t + sigma*(rand-1/2)
    void randomlyWalk(Firefly &firefly , std::uniform_real_distribution<double> score, std::mt19937 mt, double sigma) {
        for (int i = 0; i < rbfCount; i++) {
            for (int j = 0; j < dim; j++) {
                firefly.weights(i, j) += sigma * (score(mt) - 0.5);
            }
            
            firefly.spreads[i] += sigma * (score(mt) - 0.5);
            
            for (int j = 0; j < dim; j++) {
                firefly.centerVector[i](j) += sigma * (score(mt) - 0.5);
            }
        }
        
        for (int i = 0; i < dim; i++) {
            firefly.biases[i] += sigma * (score(mt) - 0.5);
        }
    }
    
    //一番いい結果のindexで出力
    Eigen::VectorXd output(const Eigen::VectorXd &input) const {
        return fireflies[bestFireflyIndex].output(input);
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
