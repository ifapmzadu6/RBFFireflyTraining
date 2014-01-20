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


class Firefly;

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
//    std::vector<Firefly> fireflies;
    std::vector<std::shared_ptr<Firefly>> firefliesPtr;
    //fireflyが移動する時の移動係数
    double attractiveness;
    double attractivenessMin;
    //fireflyが移動する時のランダム要素の倍率係数gumma
    double gumma;
    //乱数の影響力を決める
    double alpha;
    //NS-FAのfireflyの関係図のフラグ(fireflyCount, fireflyCount),Network-Structured Firefly Algorithmで使用
    std::vector<std::vector<bool>> connection;
    //乱数生成
    std::mt19937 mt;
    std::uniform_real_distribution<double> score;
    std::normal_distribution<double> nscore;
    
public:
    //コンストラクタ
    FireflyRBFTraining(int dim, int dataCount, int rbfCount, int fireflyCount, double attractiveness, double gumma, int maxGeneration);
    
    //Fireflyの初期化
    void makeFireflyWithRandom();
    //inputのセンターベクトルからfireflyのセンターベクトルを生成する
    void makeFireflyWithInput(std::vector<std::vector<double>> &inputs);
    //データからFireflyをひとつ生成する
    void makeFireflyWithData(std::vector<std::vector<double>> &weights, std::vector<std::vector<double>> &centerVector, std::vector<double> &spreads, std::vector<double> &biases);
    
    //教師信号入力input、教師信号出力output
    void training(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs);
    
    //一番いい結果のFireflyで出力
    std::vector<double> output(const std::vector<double> &input);
    //一番いい結果のFireflyを出力
    void outputBestFirefly(std::ofstream &output);
    
private:
    //各要素の距離
    inline const double distanceBetweenTwoFireflies(const Firefly &firefly1, const Firefly &firefly2);
    //移動  firefly <- (1-beta)*firefly + beta*firefly + (rand-1/2)
    inline void moveFirefly(Firefly &firefly, const Firefly &destinationFirefly);
    //-1から1の間におさめる
    inline void findLimits(Firefly &firefly);
    //ランダムに移動  t <- t + sigma*(rand-1/2)*L　（乱数は正規分布を与える。）
    inline void randomlyWalk(Firefly &firefly);
    static bool compare(const Firefly &obj1, const Firefly &obj2);
};

#endif
