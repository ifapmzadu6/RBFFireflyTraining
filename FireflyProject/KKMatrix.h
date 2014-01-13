//
//  KKMatrix.h
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2014/01/12.
//  Copyright (c) 2014年 Keisuke Karijuku. All rights reserved.
//

#ifndef FireflyProject_KKMatrix_h
#define FireflyProject_KKMatrix_h

#include <vector>

class KKMatrix {
    
    std::vector<std::vector<double>> matrix;
    
    //コンストラクタ
    KKMatrix();
    KKMatrix(size_t rows, size_t cols);
    KKMatrix(size_t rows, size_t cols, int value);
    KKMatrix(size_t rows, size_t cols, double value);
    
    //イテレータの操作
    std::vector<std::vector<double>>::iterator rowBegin();
    std::vector<std::vector<double>>::iterator rowRevBegin();
    std::vector<std::vector<double>>::iterator colBegin();
    std::vector<std::vector<double>>::iterator colRevBegin();
    
    
};

#endif
