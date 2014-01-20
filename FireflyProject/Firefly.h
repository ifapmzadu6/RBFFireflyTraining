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
    //RBFの重み(rbfCount, dim)
    std::vector<std::vector<double>> weights;
    //RBFのシグマ(rbfCount)
    std::vector<double> spreads;
    //RBFのセンターベクトル(rbfCount, dim)
    std::vector<std::vector<double>> centerVectors;
    //outputのバイアス(rbfCount)
    std::vector<double> biases;
    
public:
    //コンストラクタ
    Firefly();
    Firefly(int dim, int dataCount, int rbfCount, const std::vector<std::vector<double>> &weights, const std::vector<double> &spreads, const std::vector<std::vector<double>> &centerVectors, const std::vector<double> &biases);
    //コピーコンストラクタ
    Firefly(const Firefly &firefly);
    //適応度を計算
    void calcFitness(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs);
    //出力
    std::vector<double> output(const std::vector<double> &input) const;
    
private:
    inline const double function(const double &spreads, const std::vector<double> &centerVector, const std::vector<double> &x) const;
    inline const double mse(const std::vector<std::vector<double>> &d, const std::vector<std::vector<double>> &o) const;
    inline const double norm(const std::vector<double> &a, const std::vector<double> &b) const;
    inline void mult(std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B);
};


#endif
