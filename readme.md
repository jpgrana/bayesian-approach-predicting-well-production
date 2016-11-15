# Bayesian Approach to Predicting Well Production

This repo contains a Bayesian analysis of predicting well production from completion parameters in a shale reservoir.  The dataset was 178 wells and 25 features, which is not made available.  The model is Bayesian hierarchical linear-regression implemented from the PyMC3 library in Python.

## Motivation

Completion parameters refer to the engineering parameters of the well such as well length, number of stages, amount of fluid, etc.  Geologic parameters are not a direct input to the model, but are handled indirectly through the hierarchical (multi-level) implementation of the linear-regression.  The grouping in this example is by reservoir zone (3 zones) but can be expanded to include field boundaries (8 fields).

## Choice of Model

An algorithm that handles non-linear effects, such as random forest or neural net would likely give a more accurate prediction.  In this case, the advantage of linear-regression is the ability to interpret the features and apply the multi-level modeling.  It also makes a nice entry point to Bayesian modeling.

## Target and Features

Choosing a metric for production of a well is always a challenge.  The data included the daily output of oil, gas and water for each well.  Oil was chosen as the fluid to predict, though gas or some combination could also be chosen.  The code includes application of a Savitzky-Golay filter to smooth the oil curve on a 31-day window and return the peak daily production as the target.  This is a common industry practice because cumulative numbers can be affected by mechanical or storage issues not related to the reservoir.

![GitHub Logo](/images/production_plot.png)

The features were reduced from 25 to 13 using a two-step process.  First, a correlation matrix was generated and one of any pair of features with correlation greater than 0.9 were removed.  Then Lasso regularization was applied to remove additional features not important to the regression.

## Pooled Model

What makes Bayesian modeling different is that instead of estimating single values for the model parameters and resulting predictions, we estimate distributions and carry the uncertainty through the modeling process.  The pooled description for this model refers to running a single regression on all the data without regard to reservoir zone or field.  If the inputs are standardized, we can interpret the importance of features based on their coefficients relative to the zero line and their spreads.

![GitHub Logo](/images/coeff_pooled.png)

We also get unique uncertainties for each prediction rather than estimating uncertainty from a uniform model average.

![GitHub Logo](/images/pred_pooled.png)

A histogram of the uncertainties for each prediction shows a range of +/- 92 to +/- 245 bbl with an average around +/- 140 bbl, based on two standard deviations.  This compares to an average of 188 bbl based on model RMSE.

![GitHub Logo](/images/pred_uncertainty.png)

## Hierarchical Model

The hierarchical model involves either part-pooling or un-pooling the data in the model so that each reservoir zone is allowed to have its own intercept.  This makes sense because at least one of the reservoir zones shows higher average production.

![GitHub Logo](/images/reservoir_violinplot.png)

The part-pooled model assumes that the intercepts themselves come from a common distribution whereas the un-pooled model assumes they have independent distributions.  The pooled model is likely to under-fit the data whereas the un-pooled model is likely to over-fit.  The part-pooled model represents a compromise between the two extremes.  This is especially helpful when one of the zones has fewer wells so we utilize the common distribution of all zones to "fill-in" missing information.  In this case, the part-pooled model give the lowest test RMSE and is the preferred model.

![GitHub Logo](/images/b0_dist_reservoir.png)
