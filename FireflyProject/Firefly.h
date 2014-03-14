//
//  Firefly.h
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2014/01/13.
//  Copyright (c) 2014年 Keisuke Karijuku. All rights reserved.
//

#ifndef FireflyProject_Firefly_h
#define FireflyProject_Firefly_h

#include <vector>
#include <random>

class Firefly
{
public:
    //教師信号input,outputの次元
    int dim;
    //教師信号input,outputのデータ数
    int dataCount;
    //RBFの数
    int rbfCount;
    //適応度
    double fitness;
    //fireflyが移動する時の移動係数
    double attractiveness;
    double attractivenessMin;
    //fireflyが移動する時のランダム要素の倍率係数gumma
    double gumma;
    
    //RBFの重み(rbfCount, dim)
    std::vector<std::vector<double>> weights;
    //RBFの係数（上に凸か下に凸か決める）
    std::vector<double> alphas;
    //RBFのシグマ(rbfCount)
    std::vector<double> spreads;
    //RBFのセンターベクトル(rbfCount, dim)
    std::vector<std::vector<double>> centerVectors;
    //RBFの出力バイアス(rbfCount)
    std::vector<double> biases;
    
public:
    //コンストラクタ
    Firefly();
    Firefly(int dim, int dataCount, int rbfCount, double attractiveness, double attractivenessMin, double gumma, const std::vector<std::vector<double>> &weights, const std::vector<double> &spreads, const std::vector<std::vector<double>> &centerVectors, const std::vector<double> &biases);
    //コピーコンストラクタ
    Firefly(const Firefly &firefly);
    
    //適応度を計算
    void calcFitness(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs, std::mt19937 &mt, std::uniform_real_distribution<double> &score);
    //距離
    const double normToFirefly(const Firefly &firefly) const;
    //移動  firefly <- (1-beta)*firefly + beta*firefly + (rand-1/2) （乱数は一様分布を与える）
    void moveToFirefly(const Firefly &firefly, double &alpha, std::mt19937 &mt, std::uniform_real_distribution<double> &score);
    //ランダムに移動  t <- t + alpha*(rand-1/2)*L　（乱数は正規分布を与える）
    void randomlyWalk(double &alpha, std::mt19937 &mt, std::uniform_real_distribution<double> &score);
    //上限、下限に値をおさめる
    void findLimits();
        
    //出力
    std::vector<double> output(const std::vector<double> &input) const;
    
private:
    const double function(const double &spreads, const std::vector<double> &centerVector, const std::vector<double> &x) const;
    const double mse(const std::vector<std::vector<double>> &d, const std::vector<std::vector<double>> &o) const;
    const double squeredNorm(const std::vector<double> &a, const std::vector<double> &b) const;
    void mult(std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B) const;
};


#endif
