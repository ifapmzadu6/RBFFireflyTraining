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
#include "FireflyRBFTraining.h"
#include "Wave.h"
#include "analysis.h"
#include "KKMatrix.h"

double weights[][3] = {{-0.305027, 0.0971567, -0.367844}, {-0.168374, -0.0695188, 0.422483}, {0.408999, 0.426238, -0.135289}, {0.299042, 0.566475, -0.0987479}, {0.288779, 0.209859, -0.340822}, {-0.114781, 1.53094, 1.14045}, {0.34379, -0.97804, 0.112161}, {0.697856, -0.488666, -0.345905}, {-0.0654348, 0.40639, 0.203997}, {-1.27312, -0.449056, 0.946447}, {-0.396839, 0.352684, -0.236259}, {-0.123669, 0.502593, -0.537179}, {-0.609315, -0.518542, 0.106263}, {-0.00627391, 0.619307, 0.0236837}, {-0.0908562, -0.14458, 0.397014}, {-0.172048, -0.0365592, -0.0481878}, {-0.334103, -0.562332, 0.0919597}, {0.165457, -0.111763, -0.491118}, {0.165515, 0.338808, -0.212554}, {-0.123109, 0.309653, 0.0139856}, {-0.420498, 0.0632495, 0.176763}, {0.95457, 0.276299, 0.103619}, {0.886284, 1.03954, 0.380155}, {0.667344, -0.332481, -0.505391}, {0.15481, 0.38323, 0.112439}, {0.0906148, -0.475528, -0.384688}, {0.14076, -0.13693, 0.28136}, {0.53659, -0.0806092, 0.857337}, {0.691417, -0.507082, 0.0895721}, {0.668149, -0.241649, 0.353674}, {0.0245127, 0.288015, -0.716752}, {-0.114324, 0.182263, 0.575759}, {-0.361776, -0.156882, 0.336526}, {0.130157, 0.228629, 0.420846}, {-0.962736, 0.358889, -0.0490939}, {-0.0419001, -0.240568, 0.597034}, {0.118962, 1.05871, 0.898039}, {-0.775549, 0.701695, 0.341993}, {-0.153824, -0.194534, 0.658539}, {-0.770154, 0.396953, -0.174753}, {-0.272179, -0.00256086, 0.379733}, {0.0164933, -0.208395, -0.118309}, {0.367978, -1.10262, -0.593384}, {-0.393814, 0.444156, 0.32006}, {-0.746645, -0.143321, 0.110834}, {1.10583, -0.270163, 0.29216}, {0.323884, 0.394095, -0.20012}, {-0.0822365, 0.194892, -0.428848}, {0.697739, -0.344306, 0.858324}, {-0.225587, -0.103378, -0.245044}, {0.360696, -0.26966, 0.168714}, {1.15408, -0.427114, 0.758868}, {0.841335, 0.74872, -0.460789}, {-0.305278, 0.297479, 0.000553952}, {0.0858244, -0.0233768, -0.649656}, {-0.629839, 0.191636, 0.15537}, {-0.64659, 0.0430523, -0.269237}, {-0.556334, -0.734187, -0.190927}, {-0.0529303, 0.291137, 0.778311}, {-0.256959, 0.284493, -0.00820218}, {0.348896, 0.780546, 0.424523}, {-0.477929, 0.00588724, -0.208262}, {-0.29511, -0.437806, -0.394934}, {-0.556955, -0.0542462, -0.103999}, {0.364071, 1.30018, 0.779774}, {-0.443009, 0.115203, -0.212123}, {0.371761, 0.673562, 0.726522}, {0.0724725, -0.330718, 0.100865}, {0.0724425, 0.248406, 0.294284}, {0.412379, -0.316958, -0.919343}, {-0.468677, -0.348835, -0.330479}, {-0.46107, -0.432321, -1.14266}, {0.68924, 0.0211282, 0.41822}, {0.657634, -0.0164376, 0.823005}, {0.21876, 0.561058, 0.251152}, {-0.264448, -0.640774, -0.505837}, {0.29904, 0.352012, -0.373501}, {0.168615, 0.4237, 0.60847}, {0.475008, -0.258147, -0.034312}, {0.0577533, 0.0420978, 0.816788}, {0.602452, 0.0382665, -1.08584}, {-0.541153, -0.619156, 0.113941}, {0.526832, 0.0275629, 0.378952}, {0.960437, 0.0569663, 0.567605}, {0.00820964, -0.361305, -0.731621}, {0.283726, 0.285138, -0.0428587}, {0.420184, 0.104429, -0.290671}, {-0.625314, 0.368011, -0.721121}, {0.105271, 0.843375, 0.169}, {0.0121642, 0.526578, 0.0877002}, {0.662576, 0.674877, 0.639051}, {0.535628, 0.670036, 0.387521}, {0.571268, 0.641368, 0.0801363}, {0.130642, -0.339048, 1.2124}, {0.578319, 1.20832, -0.531433}, {-0.189637, 0.879397, -0.175743}, {-0.095506, -0.202657, 0.939684}, {0.00828047, -0.118391, 0.420126}, {0.436414, -0.287378, 0.46259}, {0.0331435, -0.0553382, 0.462593}};

double centerVector[][3] = {{0.39464, 0.373918, 1.20004}, {-0.255802, 1.35194, 0.0899285}, {-0.0625916, 0.134756, 0.699162}, {0.765476, 0.0305014, 0.695326}, {0.520169, 0.534045, 0.440626}, {-1.83259, 0.639455, 0.535178}, {0.985764, 0.376037, 0.610802}, {0.505121, 0.692869, 0.0523284}, {1.09492, 0.395366, 0.301236}, {0.326742, 0.441674, 0.0757762}, {-0.0937607, -0.354118, 0.699614}, {-0.0733369, 0.522808, 0.633997}, {-0.166502, 0.646155, 1.06611}, {0.242245, 1.36263, 0.27135}, {0.0429318, 1.40487, 0.433069}, {0.884968, 1.56678, 0.667001}, {0.810303, 0.96403, 1.0637}, {0.32292, 0.0975467, -0.267759}, {1.3638, 0.977016, 0.0241337}, {0.133883, 1.14398, -0.0959892}, {0.9083, -0.257877, 1.00136}, {0.300112, 0.503409, 0.168955}, {0.199267, 0.796134, 0.198187}, {-0.335326, 0.587087, 0.728137}, {0.201702, -0.0721638, 0.262567}, {0.717107, -0.218648, 0.172687}, {0.937096, 1.09708, 0.604725}, {0.305321, -0.750026, -0.558039}, {-0.607668, 0.223861, 1.19736}, {-0.0956359, 0.63517, 0.533363}, {0.35014, 1.68312, 0.904832}, {0.218677, 0.440652, 0.48381}, {0.190807, -0.227978, 0.496005}, {-0.0741117, 0.577448, 1.22905}, {0.659544, 0.0243486, 1.10515}, {0.425175, 0.690851, -0.293783}, {0.273338, -0.851177, 1.189}, {0.417112, 0.873654, -0.455367}, {1.11242, -0.552068, 1.021}, {1.01022, 0.0631351, 0.285278}, {0.970677, -0.270121, -0.132803}, {0.17563, 0.885087, 0.754432}, {1.07006, -0.4671, 0.840833}, {0.275552, 1.1364, 0.959309}, {0.204521, 0.398477, 0.428596}, {0.433549, 0.810248, 0.985478}, {0.422251, 0.493045, 0.0131597}, {0.406951, 1.49861, -0.38119}, {0.0441539, 0.30381, -0.500161}, {0.404103, -0.298519, 1.0163}, {1.73618, 0.269147, -0.0664223}, {1.45264, -0.024424, -0.118409}, {1.46735, 1.65866, 0.0944097}, {-0.642571, 0.376697, -0.291616}, {0.827898, 0.482771, -0.122046}, {0.541129, 0.307174, 0.0845674}, {0.905375, 0.293271, 0.530294}, {0.378411, 0.379621, 0.987408}, {1.35842, 0.649318, 1.23765}, {0.991735, -0.577492, 0.624147}, {-0.173185, 0.506413, -0.51436}, {0.663036, 1.36339, 0.452025}, {0.91045, 0.839006, -0.0328078}, {0.996452, 0.109797, 0.596237}, {0.353196, 1.19237, -1.15038}, {0.519671, 0.745032, 1.19911}, {1.09757, -0.732732, 0.654136}, {0.959693, 1.41486, 0.561257}, {0.626418, 1.38891, 0.741559}, {0.279315, 0.514337, -0.495738}, {0.483973, 0.972228, 0.325773}, {1.10707, 1.07742, -0.214955}, {1.33555, 0.923794, 1.14477}, {-0.0876955, 0.345755, 1.14553}, {0.206054, -0.0183878, -0.647468}, {0.356005, 0.942783, 0.453371}, {0.779293, 0.511918, 0.472234}, {0.133878, 0.351085, -0.0083796}, {0.272598, 0.133662, 0.455301}, {0.562737, 1.18687, -0.13445}, {-0.553444, -0.300832, 1.88821}, {0.245063, 0.0532883, 0.22736}, {0.18517, 0.0665331, 0.0189034}, {0.333059, -0.454788, 1.47813}, {0.360637, 0.635041, -0.380132}, {1.18779, 0.759642, 0.701266}, {1.61781, -0.378424, 0.923812}, {0.239479, -0.148381, 0.601488}, {0.34486, -1.04344, 0.144686}, {1.31193, 0.327416, 0.408549}, {0.968281, 0.277042, 0.0834315}, {-0.207368, 2.27067, 0.492717}, {1.10269, 0.167142, 0.620992}, {-0.000127588, 1.32766, 1.49222}, {0.511063, -0.0906781, 1.17113}, {0.919775, 0.903473, 1.05527}, {-0.0849308, -0.673454, 0.425511}, {1.17382, 0.823437, 1.04479}, {1.01766, 1.35231, 0.228646}, {0.712046, 0.989915, 0.586238}};

double spreads[] = {0.643878, 0.960119, 0.310307, 1.00331, 0.16635, 0.836274, 1.42473, 1.36614, -0.0792801, 0.95978, 0.758904, 0.362034, -0.445941, 0.790929, 0.00783592, 0.445183, 1.09164, 0.424647, 1.24735, 1.42042, 1.61706, 0.0925578, 0.264132, 0.101264, 0.669531, 1.50023, 0.48525, 1.5905, 0.81872, 0.922855, 0.714218, -0.570977, 0.102765, 1.04652, 0.876813, 0.106517, 1.5868, 0.645186, 1.35566, 0.921131, 0.441073, 0.11841, 0.549984, 0.187688, 1.01563, 1.04736, -0.669183, 0.348552, 0.678213, -0.253775, 0.673955, 0.486132, 1.47489, 0.632167, 0.430744, 0.45632, 1.1604, 0.571528, 0.90091, 0.478126, 0.640674, 1.45027, 0.0362249, 1.16396, 1.21583, 1.0422, 1.11889, 0.470885, 1.45205, 0.639527, 0.779691, -0.26137, 0.819285, 0.798824, 1.06895, -0.143048, 0.544387, -0.230692, 0.926967, -0.0301841, -0.11554, 0.631295, 0.597695, 1.20634, 0.895925, 0.414832, 0.752574, 0.549688, 1.42637, 1.50668, 1.58212, 1.11014, 0.185799, 0.457298, 1.00718, 0.998264, 1.06549, 0.245001, 1.01513, 0.965969};

double biases[] = {-0.47008, -0.708924, 0.833315};

double LogisticMap(double x)
{
	return 3.7 * x * (1.0 - x);
}

int main(int argc, char *argv[])
{
    using namespace std;
    
    int dim = 1;
    int dataCount = 500;
    int rbfCount = 200;
    int fireflyCount = 500;
    int maxGeneration = 20000;
    
    int offset = 1;
    int startDataCount = 0;
    
//    random_device random;
//    mt19937 mt(random());
//    uniform_real_distribution<double> score(0.0, dataCount - dim * offset);
    
    vector<double> tmp;
	vector<int> fp;
	Wave wav;
	if(wav.InputWave("sample.wav") != 0)
		return -1;
	wav.StereoToMono();
	wav.GetData(tmp);
	GetFlucPeriod(fp, tmp);
//    std::cout << fp.size() << std::endl;
    
//    double max = 0, min = 99999999;
//	for (int i = startDataCount; i < dataCount + offset; i++) {
//		if (max < fp[i]) max = fp[i];
//		if (min > fp[i]) min = fp[i];
//	}
//	std::vector<double> ffp(fp.size());
//	for (int i = startDataCount; i < dataCount + offset; i++) {
//		ffp[i] = ((double)fp[i] - min) / (max - min);
//	}
    
    vector<vector<double> > input;
    vector<vector<double> > output;
    
//    for (int i = 0; i < dataCount; i++) {
//        vector<double> tmpVector(dim);
////		for (int j = 0; j < dim; j++) tmpVector[j] = tmp[startDataCount + i + j * offset];
//        for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset];
//		input.push_back(tmpVector);
//        
////        for (int j = 0; j < dim; j++) tmpVector[j] = tmp[startDataCount + i + j * offset + 1];
//		for (int j = 0; j < dim; j++) tmpVector[j] = ffp[startDataCount + i + j * offset + 1];
//		output.push_back(tmpVector);
//	}
    
//    //ロジスティック写像を使うとき
    double x = 0.01;
    vector<double> tmpVector(1);
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
    
    
    //データからFireflyを作成
//    std::vector<std::vector<double>> w;
//    std::vector<double> tmpw(dim);
//    for (int i = 0; i < rbfCount ; i++) {
//        for (int j = 0; j < dim; j++) {
//            tmpw[j] = weights[i][j];
//        }
//        w.push_back(tmpw);
//    }
//    std::vector<std::vector<double>> cv;
//    std::vector<double> tmpcv(dim);
//    for (int i = 0; i < rbfCount ; i++) {
//        for (int j = 0; j < dim; j++) {
//            tmpcv[j] = centerVector[i][j];
//        }
//        cv.push_back(tmpcv);
//    }
//    std::vector<double> s;
//    for (int i = 0; i < rbfCount; i++) {
//        s.push_back(spreads[i]);
//    }
//    std::vector<double> b;
//    for (int i = 0; i < dim; i++) {
//        b.push_back(biases[i]);
//    }
    
    FireflyRBFTraining fireflyRBF(dim, dataCount, rbfCount, fireflyCount, 1.0, 0.1, maxGeneration);
    fireflyRBF.makeFireflyWithRandom();
//    fireflyRBF.makeFireflyWithData(w, cv, s, b);
    
    fireflyRBF.training(input, output);
    
    ofstream fireflyOfs("bestFirefly.txt");
    fireflyRBF.outputBestFirefly(fireflyOfs);
    fireflyOfs.close();
    
    ofstream ofs("temp.txt");
    
//    vector<double> tmpInput(dim);
//    tmpInput = input[0];
//    for (int i = 0; i < 100; i++) {
//        ofs << i << " " << output[i+1][0] << " ";
//        tmpInput = fireflyRBF.output(tmpInput);
//        ofs << tmpInput[0];
//        ofs << endl;
//    };
    
//    //ロジスティック写像を使うとき
    for (double x = 0; x < 1.0; x += 0.01) {
        tmpVector[0] = x;
        ofs << x << " " << LogisticMap(x) << " " << fireflyRBF.output(tmpVector)[0] << std::endl;
    }
    
    ofs.close();
    
    std::system("/opt/local/bin/gnuplot -persist -e \" p 'temp.txt' u 1:2 w l, '' u 1:3 w lp \"");
    
    return 0;
}
