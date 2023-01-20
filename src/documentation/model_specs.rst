.. _model_specifications:

********************
Model specifications
********************

The directory *src.model_specs* contains `JSON <http://www.json.org/>`_ files with model specifications.
These files define the different model specifications used to calculate synthetic controls for different szenarios.

You can input the treatment country, the treatment date, the period you are considering, the donor pool, the outcome variable and all predictor variables.

Thereby everything except the predictor variables are defined in the files named synth_control_parameters_*specification_name*, and the predictors in predictors_*specification_name*. In the later one also the names of the predictors to input in the table are defined.
It is important that for every specification both are specified.

These specifications are, as explained in the introduction:

1. a baseline specification for GDPPC: outcome is GDPPC and predictors are yearly averages of the outcome for 2008, 2010, 2012, 2013, 2014
2. a all-year averages specification for GDPPC: outcome is GDPPC and predictors are yearly averages of the outcome for 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014
3. a with-covariates specification: outcome is GDPPC and predictors are an averages of the outcome for 2012, 2013, 2014 and other covariates: trade opennes, inflation, industry share and schooling
4. a current-account specification: outcome is CA as % of GDP and predictors are an averages of the outcome for 2012, 2013, 2014 and other covariates: trade opennes, inflation, industry share and schooling
5. a time placebo specification: as baseline, but treatment time changes to 2005Q1 and accordingly sample period and predictor years are ten years earlier
6. a country-placebo specification: as baseline, but treated country changes to Australia and Switzerland is excluded
