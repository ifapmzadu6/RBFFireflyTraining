//
//  main.cpp
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2013/10/21.
//  Copyright (c) 2013年 Keisuke Karijuku. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "FireflyRBFTraining.h"
#include "Wave.h"
#include "analysis.h"

double LogisticMap(double x)
{
	return 3.7 * x * (1.0 - x);
}

int main(int argc, char *argv[])
{
    using namespace std;
    using namespace Eigen;
    
    int dim = 1;
    int dataCount = 1200;
    int offset = 7;
    int rbfCount = 100;
    int fireflyCount = 100;
    int iteration = 40;
    
    int startDataCount = 100000;
    
    //many thanks for your kind attention.
    
    vector<double> tmp;
	vector<int> fp;
	Wave wav;
	if(wav.InputWave("sample.wav") != 0)
		return -1;
	wav.StereoToMono();
	wav.GetData(tmp);
//	GetFlucPeriod(fp, tmp);
//    double max = 0, min = 99999999;
//	for (int i = 0; i < dataCount + offset; i++) {
//		if (max < fp[i]) {
//			max = fp[i];
//		}
//		if (min > fp[i]) {
//			min = fp[i];
//		}
//	}
//    double max = 0, min = 99999999;
//	for (int i = 0; i < dataCount + offset; i++) {
//		if (max < tmp[i]) max = fp[i];
//		if (min > tmp[i]) min = fp[i];
//	}
//	std::vector<double> ffp(fp.size());
//	for (int i = 0; i < dataCount + offset; i++) {
//		ffp[i] = (fp[i] - min) / (max - min);
//	}
    
    vector<VectorXd> input;
    vector<VectorXd> output;
    
//    random_device random;
//    mt19937 mt(random());
//    uniform_real_distribution<double> score(0.0, dataCount - dim * offset);
    
    VectorXd xv(dim), yv(dim);
    for (int i = 0; i < dataCount; i++) {
//        int rand = score(mt);
		for (int j = 0; j < dim; j++) {
			xv[j] = tmp[startDataCount + i + j * offset];
		}
		input.push_back(xv);
        
		for (int j = 0; j < dim; j++) {
			yv[j] = tmp[startDataCount + i + j * offset + 1];
		}
		output.push_back(yv);
	}
    
//    for (int i = 0; i < dataCount; i++) {
//		Eigen::VectorXd xv(dim), yv(dim);
//		for (int j = 0; j < dim; j++) {
//			xv[j] = ffp[i + j * offset];
//		}
//		input.push_back(xv);
//        
//		for (int j = 0; j < dim; j++) {
//			yv[j] = ffp[i + j * offset + 1];
//		}
//		output.push_back(yv);
//	}
    
    //ロジスティック写像を使うとき
//    double x = 0.01;
//    Eigen::VectorXd tmpVector(1);
//    tmpVector(0) = x;
//    input.push_back(tmpVector);
//    for (int i = 0; i < dataCount; i++) {
//        x = LogisticMap(x);
//        tmpVector(0) = x;
//        if (i < dataCount - 1) {
//            input.push_back(tmpVector);
//        }
//        output.push_back(tmpVector);
//    }
    
    FireflyRBFTraining fireflyRBF(dim, dataCount, rbfCount, fireflyCount, 1.0, 0.1, iteration);
    
    fireflyRBF.training(input, output);
    
    ofstream ofs("temp.txt");
    
//    for (int i = 0; i < 100; i++) {
//        ofs << i << " " << fp[i] << std::endl;
//    }
    
    VectorXd tmpInput(dim);
    for (int i = 0; i < dim; i++) {
        tmpInput(i) = input[0](i);
    }
    for (int i = 0; i < 1000; i++) {
        ofs << i << " " << output[i](0) << " ";
//        tmpInput = fireflyRBF.output(tmpInput);
//        ofs << tmpInput(0);
        ofs << endl;
    };
    
//    for (double x = 0; x < 1.0; x += 0.01) {
//        Eigen::VectorXd tmpVector(1);
//        tmpVector[0] = x;
//        ofs << x << " " << LogisticMap(x) << " " << fireflyRBF.output(tmpVector) << std::endl;
//    }
    
    ofs.close();
    
    std::system("/opt/local/bin/gnuplot -persist -e \" p 'temp.txt' u 1:2 w l, '' u 1:3 w lp \"");
    
    return 0;
}
