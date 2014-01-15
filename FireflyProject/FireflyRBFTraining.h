//
//  FireflyBRFTraining.h
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2013/10/23.
//  Copyright (c) 2013年 Keisuke Karijuku. All rights reserved.
//

#ifndef FireflyProject_FireflyBRFTraining_h
#define FireflyProject_FireflyBRFTraining_h

#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <sys/time.h>

#include "Firefly.h"

class FireflyRBFTraining
{    
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
    std::vector<std::vector<bool> > connection;
    //乱数生成
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
    }
    
    //Fireflyの初期化
    void makeFireflyWithRandom();
    void makeFireflyWithData(std::vector<std::vector<double>> &weights, std::vector<std::vector<double>> &centerVector, std::vector<double> &spreads, std::vector<double> &biases);
    
    //教師信号入力input、教師信号出力output
    void training(const std::vector<std::vector<double> > &input, const std::vector<std::vector<double> > &output);
    
    //一番いい結果のFireflyで出力
    std::vector<double> output(const std::vector<double> &input);
    //一番いい結果のFireflyを出力
    void outputBestFirefly(std::ofstream &output);
    
private:
    //各要素の距離
    inline double distanceBetweenTwoFireflies(const Firefly &firefly1, const Firefly &firefly2);
    //移動  firefly <- (1-beta)*firefly + beta*firefly + (rand-1/2)
    inline void moveFirefly(Firefly &firefly, const Firefly &destinationFirefly);
    //-1から1の間におさめる
    inline void findLimits();
    //ランダムに移動  t <- t + sigma*(rand-1/2)*L
    inline void randomlyWalk(Firefly &firefly);
    static bool compare(const Firefly &obj1, const Firefly &obj2) {
        return obj1.fitness < obj2.fitness;
    }
};

#endif
