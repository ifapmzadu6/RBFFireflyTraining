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
#include <cmath>


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
private:
    //RBFのoutput(dataCount, rbfCount)
    std::vector<std::vector<double>> tmpRbfOutput;
    //output(dataCount, dim)
    std::vector<std::vector<double>> tmpOutput;
    //tmp(rbfCount)
    std::vector<double> tmpVector;
    //tmp(dim)
    std::vector<double> tmpVector1;
    
public:
    //コンストラクタ
    Firefly();
    Firefly(int dim, int dataCount, int rbfCount, const std::vector<std::vector<double>> &weights, const std::vector<double> &spreads, const std::vector<std::vector<double>> &centerVectors, const std::vector<double> &biases);
    
    inline double norm(const std::vector<double> &a, const std::vector<double> &b) const;
    inline void mult(std::vector<std::vector<double>> &Y, const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B);
    
    inline double function(const double &spreads, const std::vector<double> &centerVector, const std::vector<double> &x) const;
    inline double mse(const std::vector<std::vector<double>> &d, const std::vector<std::vector<double>> &o) const;
    inline void calcRbfOutput(const std::vector<std::vector<double>> &inputs);
    void calcFitness(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs);
    
    std::vector<double> output(const std::vector<double> &input);
};


#endif
