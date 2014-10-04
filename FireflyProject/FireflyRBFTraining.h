//
//  FireflyBRFTraining.h
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2013/10/23.
//  Copyright (c) 2013年 Keisuke Karijuku. All rights reserved.
//

#ifndef FireflyProject_FireflyBRFTraining_h
#define FireflyProject_FireflyBRFTraining_h

#include <vector>
#include <random>
#include <fstream>

#include "Firefly.h"

class FireflyRBFTraining
{
private:
    //最大試行回数
    int maxGeneration;
    //教師信号input,outputの次元
    int dim;
    //RBFの数
    int rbfCount;
    //firefly
	int fireflyCount;
    std::vector<std::shared_ptr<Firefly>> firefliesPtr;
    //fireflyが移動する時の移動係数
    double attractiveness;
    double attractivenessMin;
    //fireflyが移動する時のランダム要素の倍率係数gumma
    double gumma;
    //乱数の影響力を決める
    double alpha;
//    //NS-FAのfireflyの関係図のフラグ(fireflyCount, fireflyCount),Network-Structured Firefly Algorithmで使用
//    std::vector<std::vector<bool>> connection;
    
public:
    //コンストラクタ
    FireflyRBFTraining(int dim, int rbfCount, int fireflyCount, double attractiveness, double gumma, int maxGeneration);
    
    //Fireflyの初期化
    void makeFireflyWithRandom();
    //inputのセンターベクトルからfireflyのセンターベクトルを生成する
    void makeFireflyWithInput(const std::vector<std::vector<double>> &inputs);
    //データからFireflyをひとつ生成する
    void makeFireflyWithData(const std::vector<std::vector<double>> &weights,const std::vector<std::vector<double>> &centerVector, const std::vector<double> &spreads, const std::vector<double> &biases);
    
    //教師信号入力input、教師信号出力output
    void training(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs);
    
    //一番いい結果のFireflyで出力
    std::vector<double> output(const std::vector<double> &input);
    //一番いい結果のFireflyを出力
    void outputBestFirefly(std::ofstream &output);    
};

#endif
