//
//  File.h
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2014/02/05.
//  Copyright (c) 2014年 Keisuke Karijuku. All rights reserved.
//

#ifndef __FireflyProject__File__
#define __FireflyProject__File__

#include <iostream>

#endif /* defined(__FireflyProject__File__) */

// binary-inputs denoising autoencoder (tied W, W' = W^T)
//  y = sigmoid(W x' + b)
//  z = sigmoid(W' y + b') ~ x
class DenoisingAutoEncoder {
    int num_visible, num_hidden;
    Matrix W, bvis, bhid;
    double p, eps;
    
    double sigmoid(double x){
        return 1.0 / (1.0 + exp(-x));
    }
    
public:
    DenoisingAutoEncoder(int num_visible, int num_hidden, double p, double eps):
    num_visible(num_visible),
    num_hidden(num_hidden),
    W(num_hidden, num_visible),
    bvis(num_visible, 1),
    bhid(num_hidden, 1),
    p(p),
    eps(eps)
    {
        //initial W
        //20140115追記：num_visibleとnum_hiddenが逆
        for(int i=0; i<num_hidden; i++){
            for(int j=0; j<num_visible; j++){
                W.setVal(i, j, 1.0 - 2.0 * frand());
            }
        }
    }
    
    void train(const std::vector<int>& in){
        //make input x
        Matrix x(num_visible, 1);
        for(int i=0; i<in.size(); i++){
            x.setVal(i, 0, in[i]);
        }
        //noise addition
        Matrix noisex(num_visible, 1);
        for(int i=0; i<in.size(); i++){
            double val = in[i];
            if(frand() < p){
                if(val < 0.5) val = 1.0;
                else val = 0.0;
            }
            noisex.setVal(i, 0, val);
        }
        //vector one
        Matrix one(num_hidden, 1);
        for(int i=0; i<num_hidden; i++){
            one.setVal(i, 0, 1.0);
        }
        //make y
        Matrix y = W * noisex + bhid;
        for(int i=0; i<num_hidden; i++){
            y.setVal(i, 0, sigmoid(y.getVal(i, 0)));
        }
        //make z
        Matrix z = W.transpose() * y + bvis;
        for(int i=0; i<num_visible; i++){
            z.setVal(i, 0, sigmoid(z.getVal(i, 0)));
        }
        
        //make grad bhid
        Matrix gradbhid = W * (x-z) * y * (one-y);
        //make grad bvis
        Matrix gradbvis = x - z;
        //make grad W
        Matrix gradW = (gradbhid * noisex.transpose()) + ((gradbvis * y.transpose()).transpose());
        
        //update W
        for(int i=0; i<num_hidden; i++){
            for(int j=0; j<num_visible; j++){
                W.setVal(i, j, W.getVal(i, j) + eps * gradW.getVal(i, j));
            }
        }
        
        //update b
        for(int i=0; i<num_hidden; i++){
            bhid.setVal(i, 0, bhid.getVal(i, 0) + eps * gradbhid.getVal(i, 0));
        }
        
        //update b'
        for(int i=0; i<num_visible; i++){
            bvis.setVal(i, 0, bvis.getVal(i, 0) + eps * gradbvis.getVal(i, 0));
        }
    }
    
    //隠れ層の値を取得
    std::vector<double> getHiddenValues(const std::vector<int>& in){
        std::vector<double> ret;
        
        //make input x
        Matrix x(num_visible, 1);
        for(int i=0; i<in.size(); i++){
            x.setVal(i, 0, in[i]);
        }
        //make y
        Matrix y = W * x + bhid;
        for(int i=0; i<num_hidden; i++){
            ret.push_back( sigmoid(y.getVal(i, 0)) );
        }
        
        return ret;
    }
    
    //出力層の値を取得
    std::vector<double> getOutputValues(const std::vector<int>& in){
        std::vector<double> ret;
        
        //make input x
        Matrix x(num_visible, 1);
        for(int i=0; i<in.size(); i++){
            x.setVal(i, 0, in[i]);
        }
        //make y
        Matrix y = W * x + bhid;
        for(int i=0; i<num_hidden; i++){
            y.setVal(i, 0, sigmoid(y.getVal(i, 0)));
        }
        //make z
        Matrix z = W.transpose() * y + bvis;
        for(int i=0; i<num_visible; i++){
            ret.push_back( sigmoid(z.getVal(i, 0)) );
        }
        
        return ret;
    }
    
    //重み行列Wを出力
    void dumpW(){
        std::cout << "Weight W" << std::endl;
        std::cout << W << std::endl;
    }
    //バイアスbとb'を出力
    void dumpb(){
        std::cout << "bhid" << std::endl;
        std::cout << bhid << std::endl;
        
        std::cout << "bvis" << std::endl;
        std::cout << bvis << std::endl;
    }
};

int main(){
    
    int InputDim, HiddenDim, Epoch;
    double noiseP, epsilon;
    int dataNum;
    
    //入力は以下の形式
    //  入力次元 隠れ層次元 反復回数 ノイズ付与の割合 学習率
    //  入力データ数
    //  入力データ...
    std::cin >> InputDim >> HiddenDim >> Epoch;
    std::cin >> noiseP >> epsilon;
    std::cin >> dataNum;
    
    DenoisingAutoEncoder da(InputDim, HiddenDim, noiseP, epsilon);
    
    //inputs
    std::vector< std::vector<int> > in(dataNum, std::vector<int>(InputDim));
    for(int i=0; i<dataNum; i++){
        for(int j=0; j<InputDim; j++){
            std::cin >> in[i][j];
        }
    }
    
    
    //train
    for(int t=0; t<Epoch; t++){
        for(int i=0; i<dataNum; i++){
            da.train(in[i]);
        }
        
        int errSum = 0;
        for(int i=0; i<dataNum; i++){
            std::vector<double> res = da.getOutputValues(in[i]);
            for(int j=0; j<InputDim; j++){
                if( in[i][j] != (res[j]>0.5?1:0) ) errSum++;
            }
        }
        
        std::cout << "epoch " << t << " : Err = " << (100.0 * errSum / (dataNum * InputDim)) << "%" << std::endl;
    }
    
    
    //results
    da.dumpW();
    da.dumpb();
    
    for(int i=0; i<dataNum; i++){
        std::vector<double> resH = da.getHiddenValues(in[i]);
        std::vector<double> resO = da.getOutputValues(in[i]);
        
        //Inputs
        std::cout << "Input:  ";
        for(int j=0; j<InputDim; j++){
            std::cout << in[i][j] << " ";
        }
        std::cout << std::endl;
        
        //Outputs
        std::cout << "Output: ";
        for(int j=0; j<InputDim; j++){
            std::cout << ((resO[j]>0.5)?1:0) << " ";
        }
        std::cout << std::endl;
        
        //Hidden Layers
        std::cout << "Hidden: ";
        for(int j=0; j<HiddenDim; j++){
            std::cout << ((resH[j]>0.5)?1:0) << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}