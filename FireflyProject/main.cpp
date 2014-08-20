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



double weights[][1] = {{0.00556309}, {-0.0249317}, {0.00446983}, {0.0238103}, {0.0353566}, {-0.0463472}, {-0.0282383}, {-0.0395023}, {-0.00392849}, {-0.0126815}, {-0.00464685}, {-0.0152898}, {-0.0539574}, {-0.0120487}, {0.023926}, {0.010101}, {0.0314147}, {0.00233643}, {-0.0126376}, {-0.0561474}, {-0.0114801}, {0.0405572}, {0.0153763}, {0.0494288}, {-0.0153098}, {-0.0243533}, {-0.0192705}, {0.0209923}, {-0.00728619}, {0.000241982}, {0.0211153}, {-0.0344933}, {0.0055287}, {-0.0847183}, {0.0329345}, {0.00525381}, {-0.00124345}, {0.00785143}, {0.000783225}, {0.00865993}, {0.0332298}, {0.00952027}, {-0.0167086}, {-0.0320322}, {0.0100778}, {-0.0242824}, {0.0211575}, {0.00227915}, {0.0085684}, {0.0559276}, {0.00911253}, {-0.00230233}, {0.0395199}, {-0.0265566}, {0.0484323}, {-0.0111943}, {0.0235543}, {-0.0459556}, {0.0040158}, {0.00504838}, {-1.40297e-05}, {-0.0258764}, {-0.0325469}, {-0.0390435}, {-0.0181758}, {-0.0269964}, {-0.0163452}, {0.0603413}, {-0.068544}, {0.00382162}, {-0.0303768}, {-0.0175179}, {-0.0084443}, {0.00909174}, {-0.00243789}, {-0.00334941}, {-0.0215885}, {-0.00222662}, {0.0278673}, {0.0173519}, {0.00344039}, {-0.0172022}, {-0.0296545}, {0.000734314}, {-0.0328335}, {-0.0288957}, {-0.0252489}, {-0.00486846}, {-0.0310428}, {0.00202007}, {-0.0190332}, {0.0113163}, {0.0126095}, {0.0129714}, {0.0111463}, {-0.0497877}, {-0.0490554}, {0.000625406}, {-0.0220863}, {0.0138684}};
double centerVector[][1] = {{0.430096}, {0.489621}, {0.50555}, {0.507019}, {0.504831}, {0.480938}, {0.496781}, {0.476326}, {0.498436}, {0.508332}, {0.448457}, {0.470451}, {0.470805}, {0.506028}, {0.491489}, {0.535678}, {0.475347}, {0.466375}, {0.484474}, {0.524556}, {0.553366}, {0.468957}, {0.52799}, {0.512416}, {0.490416}, {0.494247}, {0.473718}, {0.519718}, {0.522547}, {0.510064}, {0.51099}, {0.488308}, {0.495665}, {0.489584}, {0.509777}, {0.493975}, {0.489579}, {0.506802}, {0.488304}, {0.4869}, {0.49642}, {0.503474}, {0.496199}, {0.506806}, {0.486864}, {0.490216}, {0.550229}, {0.49805}, {0.478898}, {0.517409}, {0.418812}, {0.544465}, {0.509022}, {0.508095}, {0.531421}, {0.500665}, {0.529163}, {0.482618}, {0.483517}, {0.495859}, {0.540113}, {0.454348}, {0.507509}, {0.492796}, {0.49585}, {0.476628}, {0.46392}, {0.523888}, {0.476805}, {0.485286}, {0.530615}, {0.520454}, {0.534928}, {0.480201}, {0.490721}, {0.47546}, {0.49831}, {0.47662}, {0.547639}, {0.476294}, {0.50648}, {0.518475}, {0.516401}, {0.515402}, {0.519623}, {0.499742}, {0.513427}, {0.507729}, {0.488231}, {0.516411}, {0.526957}, {0.527859}, {0.524224}, {0.500232}, {0.516309}, {0.445414}, {0.531614}, {0.514039}, {0.48337}, {0.514738}};
double spreads[] = {0, -1, 0, 0, -1, -1, -1, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, 0};
double biases[] = {0.0470473};
double alphas[] = {-1, -0.999975, -0.999996, -1, -1, -0.999991, -0.999998, -0.837145, -1, -1, -1, -0.999994, -0.837116, -0.999984, -1, -1, -1, -0.999988, -0.999992, -1, -0.99998, -0.999995, -0.999988, -0.999987, -0.999992, -0.999992, -0.999999, -0.999996, -1, -1, -1, -0.999989, -1, -1, -1, -1, -0.999966, -0.99999, -1, -0.999999, -1, -1, -0.999999, -0.999984, -0.999993, -0.837126, -1, -1, -0.999998, -0.999998, -1, -1, -1, -1, -0.99998, -0.999989, -1, -0.999995, -0.999994, -0.999999, -1, -0.837106, -1, -1, -0.999995, -1, -0.999983, -1, -1, -0.999988, -1, -1, -1, -1, -0.999996, -0.999975, -0.999993, -1, -1, -0.999995, -0.837121, -0.999992, -0.999988, -0.999993, -1, -0.999992, -1, -0.999984, -0.999996, -1, -1, -1, -1, -1, -1, -0.837131, -0.999996, -1, -0.999983, -0.999981};





double LogisticMap(double x)
{
	return 3.7 * x * (1.0 - x);
}

void useLogisticMap();

int main(int argc, char *argv[])
{
    using namespace std;
    
    useLogisticMap();
    
//    int dim = 1;
//    int dataCount = 250;
//    int rbfCount = 100;
//    int fireflyCount = 1000;
//    int maxGeneration = 1000;
//    int offset = 1;
//    int startDataCount = 100;
//    
////    random_device random;
////    mt19937 mt(random());
////    uniform_real_distribution<double> score(0.0, dataCount - dim * offset);
//    
//    vector<double> tmp;
//	vector<int> fp;
//	Wave wav;
//	if(wav.InputWave("/Users/KeisukeKarijuku/Library/Developer/Xcode/DerivedData/FireflyProject-atoglxsngafbinaeredqypibrnpk/Build/Products/Release/sample.wav") != 0)
//		return -1;
//	wav.StereoToMono();
//	wav.GetData(tmp);
//	GetFlucPeriod(fp, tmp);
////    std::cout << fp.size() << std::endl;
//    
//    double max = 0, min = std::numeric_limits<double>::max();
//	for (int i = startDataCount; i < startDataCount + dataCount + offset; i++) {
//		if (max < fp[i]) max = fp[i];
//		if (min > fp[i]) min = fp[i];
//	}
//	std::vector<double> ffp(fp.size());
//	for (int i = startDataCount; i < startDataCount + dataCount + offset; i++) {
//		ffp[i] = ((double)fp[i] - min) / (max - min);
//	}
//    
//    vector<vector<double>> input;
//    vector<vector<double>> output;
//    for (int i = 0; i < dataCount; i++) {
//        vector<double> tmpVector(dim);
//        for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset];
//		input.push_back(tmpVector);
//		for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset + 1];
//		output.push_back(tmpVector);
//	}
//    
//    
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
//
//    FireflyRBFTraining fireflyRBF(dim, dataCount, rbfCount, fireflyCount, 1.0, 0.1, maxGeneration);
//    fireflyRBF.makeFireflyWithRandom();
////    fireflyRBF.makeFireflyWithInput(input);
//    fireflyRBF.training(input, output);
//    fireflyRBF.makeFireflyWithData(w, cv, s, b);
//
//    ofstream fireflyOfs("bestFirefly.txt");
//    fireflyRBF.outputBestFirefly(fireflyOfs);
//    fireflyOfs.close();
//
//    ofstream ofs("temp.txt");
//    vector<double> tmpInput(dim);
//    vector<double> tmpInputPushed(dim);
//    vector<double> tInput(dim);
//    tmpInput = input[0];
//    tmpInputPushed = input[0];
//    for (int i = 0; i < 100; i++) {
//        ofs << i << " " << output[i][0] << " ";
////        tmpInput = fireflyRBF.output(tmpInputPushed);
////        for (int i = 0; i < dim - 1; i++) tmpInputPushed[i] = tmpInputPushed[i+1];
////        tmpInputPushed[dim - 1] = tmpInput[dim - 1];
//        
//        tmpInput = fireflyRBF.output(tmpInput);
//        
//        ofs << tmpInput[0] << " ";
//        tInput = fireflyRBF.output(input[i]);
//        ofs << tInput[0];
//        ofs << endl;
//    };
//    ofs.close();
//    
//    std::system("/opt/local/bin/gnuplot -persist -e \" p 'temp.txt' u 1:2 w l, '' u 1:3 w lp, '' u 1:4 w l \"");
    
    return 0;
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
    
    FireflyRBFTraining fireflyRBF(dim, dataCount, rbfCount, fireflyCount, 1.0, 0.1, maxGeneration);
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