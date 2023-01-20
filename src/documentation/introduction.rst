.. _introduction:


************
Introduction
************

This project estimates the real consequences of the Swiss appreciation from January 2015 using the synthetic control method. The idea is to construct a synthetic Switzerland
as a control unit, which describes how Switzerland would have developed in absence of a large appreciation. The difference in development after the decision of the SNB to unpeg the franc
between the actual and the synthetic Switzerland is the treatment effect.

The method is based on Abadie et al. (2010, 2014). :cite:`Abadie2010` :cite:`Abadie2014`
On the basis of pre-treatment data a weighted average of control countries is constructed to work as a synthetic control for the treated country.
The method is described shortly below.

The algorithm used to calculate the synthetic control is based on Becker and Klößner (2018) :cite:`Becker2018`.
The implementation and algorithm suggested by them is explained briefly below.

The project is based on the Templates for Reproducible Research Projects in Economics by von Gaudecker. :cite:`GaudeckerEconProjectTemplates`

Structure
==========

The project is structured in different folders. Every folder contains files with dictinct functions.
The folders are:

  * original_data
  * data_management
  * analysis
  * final
  * paper
  * model_code
  * model_specs

  Following the folders, the project is organised in different steps:

  1. data management: take the raw data from **original_data** and clean it and bring it in the structure we need (second normal form)
  2. analysis: Taking the functions defining the "model", in my case implementing the algorithm of Becker and Klößner (2018) :cite:`Becker2018` from **model_code**, and the specifications defined for the models in **model_specs**, the main analysis is done. The synthetic control unit and all necessary values are calculated and the results are stored.
  3. final: In this step the results from the analysis step are visualised. This means tables and figures are constructed.
  4. paper: This is the last step. Figures and tables are incorporated and the paper is compiled.

Detailed discribtions of what exactly every step and file is doing can be found in the following chapters.

The project runs 6 different specifications contained in **model_specs**. These are:

1. a baseline specification for GDPPC: outcome is GDPPC and predictors are yearly averages of the outcome for 2008, 2010, 2012, 2013, 2014
2. a all-year averages specification for GDPPC: outcome is GDPPC and predictors are yearly averages of the outcome for 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014
3. a with-covariates specification: outcome is GDPPC and predictors are an averages of the outcome for 2012, 2013, 2014 and other covariates: trade opennes, inflation, industry share and schooling
4. a current-account specification: outcome is CA as % of GDP and predictors are an averages of the outcome for 2012, 2013, 2014 and other covariates: trade opennes, inflation, industry share and schooling
5. a time placebo specification: as baseline, but treatment time changes to 2005Q1 and accordingly sample period and predictor years are ten years earlier
6. a country-placebo specification: as baseline, but treated country changes to Australia and Switzerland is excluded

For every specification the analysis is run and tables and figures are produced.

The Synthetic Control Method
====================================

In this section I want to give a short description of the synthetic control method. It follows mainly Abadie (2010) :cite:`Abadie2010` . For details please refer to this paper and the ones cited above.

The synthetic control method is a data-driven approach to construct a suitable control unit in comparative case studies. The idea is that a combination of control units will often provide a better comparison for the treated unit than a single control unit alone. Therefore, the synthetic control is constructed as a weighted average of so called donors, a sample of suitable control units, with non-negative weights that sum to one. These weights, collected in a vector *W*, are calculated by minimizing the difference of selected predictor variables between the treated unit and the synthetic control. The predictor variables can be linear combinations of the outcome variable prior to treatment as well as other covariates with explanatory power for the outcome of interest, which need to be measured prior to or unaffected by the treatment. In the optimization the predictors are weighted by a predictor weighting matrix *V*, which is chosen to result in optimal weights *W* that yield the lowest possible RMSPE between the outcome of the treated unit and the synthetic control prior to treatment. This structure leads to a nested optimization problem. In the inner optimization the optimal donor weights *W* are determined, which construct the synthetic control unit. These optimization depends on the predictor weights *V*, which are determined in the outer optimization to guarantee the best possible pre-treatment fit. The resulting optimal weights define the synthetic control unit. The treatment effect consists of the difference in the outcome variable after treatment between the treated unit and the synthetic control.


The Implementation and  the Algorithm
=========================================

The structure of the synthetic control method described above poses some challenges to the implementation. The problem is that the nested optimization is not only computational intensive and therefore slow with larger data sets, but it might also be quite unstable and unreliable with numerical optimizers. The reason for that is that the objective function of the outer optimization contains a minimization problem, which results in a noisy function that might be ill behaved and can fool the outer optimizer. Becker and Klößner (2018) :cite:`Becker2018` provide an algorithm that tries to reduce these problems. It starts with detecting important special cases that are easy to compute and then tries to reduce the dimension of the nested optimization problem.

The basis of Becker and Klößner's argumentation consists of some theory concerning the optimization problems that have to be solved for applying the synthetic control method. They start with separating the donor pool in sunny and shady donors. A shady donor is a control unit, whose difference in predictor values to the treated unit multiplied by :math:`\alpha` with :math:`0<\alpha<1` lies inside the convex hull of the differences of all donor units. They show that if a donor is shady, it will not be part of an optimal synthetic control unit. Furthermore, they give simple solutions in cases with no sunny donors, which means exact fit is possible, or only one sunny donor, which will then be the unique donor with positive weight. If none of these special cases occur, the algorithm tests whether the unrestricted outer optimum is feasible. This means it searches for predictor weights *V* which result in donor weights *W* that constitute the global minimum of the outer optimization problem. Only if finding such predictor weights is not possible, the nested optimization is performed. In order to do this, the dimension of the problem is reduced by excluding all shady donors. A detailed description of the algorithm can be found in Becker and Klößner (2018) :cite:`Becker2018` and Figure 2 in their paper illustrate it's structure in a simple way.

Testing
===========

I do different forms of testing:

For the functions defined in the model code that construct the algorithm explained in the last section I do unit and integration tests using pytest. For the function
putting everything together I do system tests.
These tests confirm that I implemented all formulas from Becker and Klößner correctly.

All data work was firstly done and tested using Jupyter notebooks.

Furthermore, as a kind of regression testing, I compared some results to the ones computed by the implementation of the synthetic control method in stata (the synth package by Abadie et al. (2010) :cite:`Abadie2010`)
and they are similar.
