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
    //RBFの重み(rbfCount, dim)
    std::vector<std::vector<double> > weights;
    //RBFのシグマ(rbfCount)
    std::vector<double> spreads;
    //RBFのセンターベクトル(rbfCount, dim)
    std::vector<std::vector<double> > centerVectors;
    //outputのバイアス(rbfCount)
    std::vector<double> biases;
    //RBFのoutput(dataCount, rbfCount)
    std::vector<std::vector<double> > tmpRbfOutput;
    //output(dataCount, dim)
    std::vector<std::vector<double> > tmpOutput;
    //tmp(rbfCount)
    std::vector<double> tmpVector;
    //tmp(dim)
    std::vector<double> tmpVector1;
    //適応度
    double fitness;
    
    Firefly();
    
    Firefly(int dim, int dataCount, int rbfCount, const std::vector<std::vector<double> > &weights, const std::vector<double> &spreads, const std::vector<std::vector<double> > &centerVectors, const std::vector<double> &biases)
    : dim(dim), dataCount(dataCount), rbfCount(rbfCount), weights(weights), spreads(spreads), centerVectors(centerVectors), biases(biases) {
        this->tmpRbfOutput = std::vector<std::vector<double> >();
        for (int i = 0; i < dataCount; i++) {
            std::vector<double> newVector = std::vector<double>(rbfCount, 0.0);
            this->tmpRbfOutput.push_back(newVector);
        }
        this->tmpOutput = std::vector<std::vector<double> >();
        for (int i = 0 ; i < dataCount; i++) {
            std::vector<double> newVector = std::vector<double>(dim, 0.0);
            this->tmpOutput.push_back(newVector);
        }
        
        this->tmpVector = std::vector<double>(rbfCount, 0.0);
        this->tmpVector1 = std::vector<double>(dim, 0.0);
    }
    
    double function(const double &spreads, const std::vector<double> &centerVector, const std::vector<double> &x) {
        return exp(- spreads * norm(centerVector, x));
    }
    
private:
    std::vector<double>::const_iterator no_xIter;
    std::vector<double>::const_iterator no_xIterEnd;
    std::vector<double>::const_iterator no_yIter;
public:
    double norm(const std::vector<double> &a, const std::vector<double> &b) {
        double d = 0.0;
        no_xIter = a.begin();
        no_xIterEnd = a.end();
        no_yIter = b.begin();
        while (no_xIter != no_xIterEnd) {
            d += ((*no_xIter) - (*no_yIter)) * ((*no_xIter) - (*no_yIter));
            ++no_xIter;
            ++no_yIter;
        }
        return d;
    }
    
private:
    std::vector<double>::const_iterator mu_rKIter;
    std::vector<double>::const_iterator mu_rKIterEnd;
    std::vector<std::vector<double> >::const_iterator mu_wKIter;
    std::vector<double>::iterator mu_oJIter;
    std::vector<double>::iterator mu_oJIterEnd;
    std::vector<double>::const_iterator mu_wJIter;
    std::vector<std::vector<double> >::iterator mu_oIIter;
    std::vector<std::vector<double> >::iterator mu_oIterEnd;
    std::vector<std::vector<double> >::const_iterator mu_rIIter;
public:
    void mult(std::vector<std::vector<double> > &Y, const std::vector<std::vector<double> > &A, const std::vector<std::vector<double> > &B) {
        for (auto &vec : Y) {
            for (auto &value : vec) {
                value = 0.0;
            }
        }
        
        mu_oIIter = Y.begin();
        mu_oIterEnd = Y.end();
        mu_rIIter = A.begin();
        while (mu_oIIter != mu_oIterEnd) {
            mu_rKIter = (*mu_rIIter).begin();
            mu_rKIterEnd = (*mu_rIIter).end();
            mu_wKIter = B.begin();
            while (mu_rKIter != mu_rKIterEnd) {
                mu_oJIter = (*mu_oIIter).begin();
                mu_oJIterEnd = (*mu_oIIter).end();
                mu_wJIter = (*mu_wKIter).begin();
                while (mu_oJIter != mu_oJIterEnd) {
                    (*mu_oJIter) += (*mu_rKIter) * (*mu_wJIter);
                    ++mu_oJIter;
                    ++mu_wJIter;
                }
                ++mu_rKIter;
                ++mu_wKIter;
            }
            ++mu_oIIter;
            ++mu_rIIter;
        }
    }
    
private:
    std::vector<std::vector<double> >::iterator cr_rIIter;
    std::vector<std::vector<double> >::iterator cr_rIIterEnd;
    std::vector<std::vector<double> >::const_iterator cr_iIter;
    std::vector<double>::iterator cr_rJIter;
    std::vector<double>::iterator cr_rJIterEnd;
    std::vector<std::vector<double> >::const_iterator cr_cIter;
    std::vector<double>::const_iterator cr_sIter;
public:
    void calcRbfOutput(const std::vector<std::vector<double> > &inputs) {
        cr_rIIter = tmpRbfOutput.begin();
        cr_rIIterEnd = tmpRbfOutput.end();
        cr_iIter = inputs.begin();
        while (cr_rIIter != cr_rIIterEnd) {
            cr_rJIter = (*cr_rIIter).begin();
            cr_rJIterEnd = (*cr_rIIter).end();
            cr_cIter = centerVectors.begin();
            cr_sIter = spreads.begin();
            while (cr_rJIter != cr_rJIterEnd) {
                (*cr_rJIter) = function(*cr_sIter, *cr_cIter, *cr_iIter);
                ++cr_rJIter;
                ++cr_cIter;
                ++cr_sIter;
            }
            ++cr_rIIter;
            ++cr_iIter;
        }
    }
    
private:
    std::vector<std::vector<double> >::iterator cf_oIIter;
    std::vector<std::vector<double> >::iterator cf_oIIterEnd;
    std::vector<double>::iterator cf_oJIter;
    std::vector<double>::iterator cf_oJIterEnd;
    std::vector<double>::iterator cf_bIter;
public:
    void calcFitness(const std::vector<std::vector<double> > &inputs, const std::vector<std::vector<double> > &outputs) {
        calcRbfOutput(inputs);
        mult(tmpOutput, tmpRbfOutput, weights);
        
        cf_oIIter = this->tmpOutput.begin();
        cf_oIIterEnd = this->tmpOutput.end();
        while (cf_oIIter != cf_oIIterEnd) {
            cf_oJIter = (*cf_oIIter).begin();
            cf_oJIterEnd = (*cf_oIIter).end();
            cf_bIter = biases.begin();
            while (cf_oJIter != cf_oJIterEnd) {
                (*cf_oJIter) += (*cf_bIter);
                ++cf_oJIter;
                ++cf_bIter;
            }
            ++cf_oIIter;
        }
        
        fitness = 1.0 / (1.0 + mse(this->tmpOutput, outputs));
    }
    
private:
    std::vector<std::vector<double> >::const_iterator ms_dIIter;
    std::vector<std::vector<double> >::const_iterator ms_dIIterEnd;
    std::vector<std::vector<double> >::const_iterator ms_oIIter;
    std::vector<double>::const_iterator ms_dJIter;
    std::vector<double>::const_iterator ms_dJIterEnd;
    std::vector<double>::const_iterator ms_oJIter;
public:
    double mse(const std::vector<std::vector<double> > &d, const std::vector<std::vector<double> > &o) {
        double mse = 0.0;
        ms_dIIter = d.begin();
        ms_dIIterEnd = d.end();
        ms_oIIter = o.begin();
        while (ms_dIIter != ms_dIIterEnd) {
            ms_dJIter = (*ms_dIIter).begin();
            ms_dJIterEnd = (*ms_dIIter).end();
            ms_oJIter = (*ms_oIIter).begin();
            while (ms_dJIter != ms_dJIterEnd) {
                mse += (*ms_dJIter - *ms_oJIter) * (*ms_dJIter - *ms_oJIter);
                ++ms_dJIter;
                ++ms_oJIter;
            }
            ++ms_dIIter;
            ++ms_oIIter;
        }
        mse /= d.size();
        return mse;
    }
    
    std::vector<double> output(const std::vector<double> &input) {
        auto tIter = tmpVector.begin();
        auto tIterEnd = tmpVector.end();
        auto cIter = centerVectors.begin();
        auto sIter = spreads.begin();
        while (tIter != tIterEnd) {
            (*tIter) = function(*sIter, *cIter, input);
            ++tIter;
            ++cIter;
            ++sIter;
        }
        
        for (auto &value : tmpVector1) {
            value = 0.0;
        }
        auto tVIter = tmpVector.begin();
        auto tVIterEnd = tmpVector.end();
        auto wIIter = weights.begin();
        while (tVIter != tVIterEnd) {
            auto oIter = tmpVector1.begin();
            auto oIterEnd = tmpVector1.end();
            auto wJIter = (*wIIter).begin();
            while (oIter != oIterEnd) {
                (*oIter) += (*wJIter) * (*tVIter);
                ++oIter;
                ++wJIter;
            }
            ++wIIter;
            ++tVIter;
        }
        
        auto ttIter = tmpVector1.begin();
        auto ttIterEnd = tmpVector1.end();
        auto bIter = biases.begin();
        while (ttIter != ttIterEnd) {
            (*ttIter) += (*bIter);
            ++ttIter;
            ++bIter;
        }
        
        return tmpVector1;
    }
};


#endif
