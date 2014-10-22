//
//  main.cpp
//  FireflyProject
//
//  Created by Keisuke Karijuku on 2013/10/21.
//  Copyright (c) 2013年 Keisuke Karijuku. All rights reserved.
//

//many thanks for your kind attention.


#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include "FireflyRBFTraining.h"
#include "Wave.h"
#include "analysis.h"



double LogisticMap(double x)
{
	return 3.7 * x * (1.0 - x);
}

void useLogisticMap();

int main(int argc, char *argv[])
{
    using namespace std;
    
//    useLogisticMap();
    
    int dim = 3;
    int dataCount = 500;
    int rbfCount = 50;
    int fireflyCount = 500;
    int maxGeneration = 100;
    int offset = 7;
    int startDataCount = 100;
    
    // Get sound wave.
    vector<double> tmp;
	vector<int> fp;
	Wave wav;
	if(wav.InputWave("/Users/KeisukeKarijuku/Dropbox/sample.wav") != 0)
		return -1;
	wav.StereoToMono();
	wav.GetData(tmp);
	GetFlucPeriod(fp, tmp);
    std::cout << fp.size() << std::endl;
    
    // Nomilization
    double max = 0, min = std::numeric_limits<double>::max();
	for (int i = startDataCount; i < startDataCount + dataCount + offset; i++) {
		if (max < fp[i]) max = fp[i];
		if (min > fp[i]) min = fp[i];
	}
	std::vector<double> ffp(fp.size());
	for (int i = startDataCount; i < startDataCount + dataCount + offset; i++) {
		ffp[i] = ((double)fp[i] - min) / (max - min);
	}
    
    // FireflyRBFTraining
    FireflyRBFTraining fireflyRBF(dim, rbfCount, fireflyCount, 1.0, 0.1, maxGeneration);
    fireflyRBF.makeFireflyWithRandom();
    vector<vector<double>> input;
    vector<vector<double>> output;
    
    // Start Training.
    int index = 1;
    int increasement = 50;
    while (increasement*index < dataCount) {
        input.clear();
        output.clear();
        for (int i = 0; i < increasement*index; i++) {
            vector<double> tmpVector(dim);
            for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset];
            input.push_back(tmpVector);
            for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset + 1];
            output.push_back(tmpVector);
        }
        fireflyRBF.training(input, output);
        
        index++;
    }
    
    // Output to files.
    ofstream fireflyOfs("bestFirefly.txt");
    fireflyRBF.outputBestFirefly(fireflyOfs);
    fireflyOfs.close();
    
    ofstream ofs("temp.txt");
    vector<double> tmpInput(dim);
    vector<double> tmpInputPushed(dim);
    vector<double> tInput(dim);
    tmpInput = input[0];
    tmpInputPushed = input[0];
    for (int i = 0; i < 100; i++) {
        ofs << i << " " << output[i][0] << " ";
        tmpInput = fireflyRBF.output(tmpInputPushed);
        for (int i = 0; i < dim - 1; i++) tmpInputPushed[i] = tmpInputPushed[i+1];
        tmpInputPushed[dim - 1] = tmpInput[dim - 1];
        
        tmpInput = fireflyRBF.output(tmpInput);
        
        ofs << tmpInput[0] << " ";
        tInput = fireflyRBF.output(input[i]);
        ofs << tInput[0];
        ofs << endl;
    };
    ofs.close();
    
    std::system("/usr/local/bin/gnuplot -persist -e \" p 'temp.txt' u 1:2 w l, '' u 1:3 w lp, '' u 1:4 w l \"");
    
    return 0;
}


void useWaveSoundWithFireflyData () {
    using namespace std;
    
    int dim = 3;
    int dataCount = 50;
    int rbfCount = 100;
    int fireflyCount = 1000;
    int maxGeneration = 1000;
    int offset = 7;
    int startDataCount = 100;
    
    //    random_device random;
    //    mt19937 mt(random());
    //    uniform_real_distribution<double> score(0.0, dataCount - dim * offset);
    
    vector<double> tmp;
    vector<int> fp;
    Wave wav;
    if(wav.InputWave("/Users/KeisukeKarijuku/Dropbox/sample.wav") != 0) {
        return;
    }
    wav.StereoToMono();
    wav.GetData(tmp);
    GetFlucPeriod(fp, tmp);
    //    std::cout << fp.size() << std::endl;
    
    // Nomilization
    double max = 0, min = std::numeric_limits<double>::max();
    for (int i = startDataCount; i < startDataCount + dataCount + offset; i++) {
        if (max < fp[i]) max = fp[i];
        if (min > fp[i]) min = fp[i];
    }
    std::vector<double> ffp(fp.size());
    for (int i = startDataCount; i < startDataCount + dataCount + offset; i++) {
        ffp[i] = ((double)fp[i] - min) / (max - min);
    }
    
    
    vector<vector<double>> input;
    vector<vector<double>> output;
    for (int i = 0; i < dataCount; i++) {
        vector<double> tmpVector(dim);
        for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset];
        input.push_back(tmpVector);
        for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset + 1];
        output.push_back(tmpVector);
    }
    
    
    //    //データからFireflyを作成
    //    std::vector<std::vector<double>> w;
    //    std::vector<double> tmpw(dim);
    //    for (int i = 0; i < rbfCount ; i++) {
    //        for (int j = 0; j < dim; j++) tmpw[j] = weights[i][j];
    //        w.push_back(tmpw);
    //    }
    //    std::vector<std::vector<double>> cv;
    //    std::vector<double> tmpcv(dim);
    //    for (int i = 0; i < rbfCount ; i++) {
    //        for (int j = 0; j < dim; j++) tmpcv[j] = centerVector[i][j];
    //        cv.push_back(tmpcv);
    //    }
    //    std::vector<double> s;
    //    for (int i = 0; i < rbfCount; i++) s.push_back(spreads[i]);
    //    std::vector<double> b;
    //    for (int i = 0; i < dim; i++) b.push_back(biases[i]);
    
    FireflyRBFTraining fireflyRBF(dim, rbfCount, fireflyCount, 1.0, 0.1, maxGeneration);
    fireflyRBF.makeFireflyWithRandom();
    //    fireflyRBF.makeFireflyWithInput(input);
    fireflyRBF.training(input, output);
    //    fireflyRBF.makeFireflyWithData(w, cv, s, b);
    
    ofstream fireflyOfs("bestFirefly.txt");
    fireflyRBF.outputBestFirefly(fireflyOfs);
    fireflyOfs.close();
    
    ofstream ofs("temp.txt");
    vector<double> tmpInput(dim);
    vector<double> tmpInputPushed(dim);
    vector<double> tInput(dim);
    tmpInput = input[0];
    tmpInputPushed = input[0];
    for (int i = 0; i < 100; i++) {
        ofs << i << " " << output[i][0] << " ";
        //        tmpInput = fireflyRBF.output(tmpInputPushed);
        //        for (int i = 0; i < dim - 1; i++) tmpInputPushed[i] = tmpInputPushed[i+1];
        //        tmpInputPushed[dim - 1] = tmpInput[dim - 1];
        
        tmpInput = fireflyRBF.output(tmpInput);
        
        ofs << tmpInput[0] << " ";
        tInput = fireflyRBF.output(input[i]);
        ofs << tInput[0];
        ofs << endl;
    };
    ofs.close();
    
    std::system("/opt/local/bin/gnuplot -persist -e \" p 'temp.txt' u 1:2 w l, '' u 1:3 w lp, '' u 1:4 w l \"");
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void useLogisticMap () {
    int dim = 1;
    int dataCount = 2000;
    int rbfCount = 25;
    int fireflyCount = 20;
    int maxGeneration = 3000;
    
    std::vector<std::vector<double>> input;
    std::vector<std::vector<double>> output;
    
    //ロジスティック写像を使うとき
    double x = 0.01;
    std::vector<double> tmpVector(1);
    tmpVector[0] = x;
    input.push_back(tmpVector);
    for (int i = 0; i < dataCount; i++) {
        x = LogisticMap(x);
        tmpVector[0] = x;
        if (i < dataCount - 1) {
            input.push_back(tmpVector);
        }
        output.push_back(tmpVector);
    }
    
//    //データからFireflyを作成
//    std::vector<std::vector<double>> w;
//    std::vector<double> tmpw(dim);
//    for (int i = 0; i < rbfCount ; i++) {
//        for (int j = 0; j < dim; j++) tmpw[j] = weights[i][j];
//        w.push_back(tmpw);
//    }
//    std::vector<std::vector<double>> cv;
//    std::vector<double> tmpcv(dim);
//    for (int i = 0; i < rbfCount ; i++) {
//        for (int j = 0; j < dim; j++) tmpcv[j] = centerVector[i][j];
//        cv.push_back(tmpcv);
//    }
//    std::vector<double> s;
//    for (int i = 0; i < rbfCount; i++) s.push_back(spreads[i]);
//    std::vector<double> a;
//    for (int i = 0; i < dim; i++) a.push_back(alphas[i]);
//    std::vector<double> b;
//    for (int i = 0; i < dim; i++) b.push_back(biases[i]);
    
    FireflyRBFTraining fireflyRBF(dim, rbfCount, fireflyCount, 1.0, 0.1, maxGeneration);
    fireflyRBF.makeFireflyWithRandom();
//    fireflyRBF.makeFireflyWithInput(input);
//    fireflyRBF.makeFireflyWithData(w, cv, s, a, b);
    fireflyRBF.training(input, output);
    
    std::ofstream ofs("temp.txt");
    
//    std::vector<double> tmpInput(dim);
//    std::vector<double> tmpInputPushed(dim);
//    std::vector<double> tInput(dim);
//    tmpInput = input[0];
//    tmpInputPushed = input[0];
//    for (int i = 0; i < 100; i++) {
//        ofs << i << " " << output[i][0] << " ";
//        tmpInput = fireflyRBF.output(tmpInput);
//        ofs << tmpInput[0] << " ";
//        tInput = fireflyRBF.output(input[i]);
//        ofs << tInput[0];
//        ofs << std::endl;
//    };
    
    std::ofstream fireflyOfs("bestFirefly.txt");
    fireflyRBF.outputBestFirefly(fireflyOfs);
    fireflyOfs.close();
    
    //    //ロジスティック写像を使うとき
    for (double x = 0; x < 1.0; x += 0.01) {
        tmpVector[0] = x;
        ofs << x << " " << LogisticMap(x) << " " << fireflyRBF.output(tmpVector)[0] << std::endl;
    }
    
    ofs.close();
    
    std::system("/usr/local/bin/gnuplot -persist -e \" p 'temp.txt' u 1:2 w l, '' u 1:3 w l, '' u 1:4 w l\"");
}