---
title: "Systematics and Statistics"
teaching: 90
exercises: 0
questions:
- "How do we perform statistical inference?"
- "What tools do we use?"
- "How do we visualize and interpret our results?"
objectives:
- "Learn how to construct statistical models."
- "Learn how to interpret these statistical models."
keypoints:
- "We  construct statistical models to estimate systematics in our analysis."
- "cabinetry and pyhf are the tools we use here."
---

## Introduction

>
>It's beyond the scope of this tutorial to cover a lot of statistical background needed
>here. You may find some references at the end of this episode and further references found within.
>We will cover hopefully the bare-minimum needed to give you an idea of what's going on here
>and to give an introduction to some useful tools.
>
{: .testimonial}

A statistical model $$ f(\vec{x} \vert \vec{\phi}) $$ describes the probability of the data
$$ \vec{x} $$ given model paramters $$ \vec{\phi} $$. 

[HistFactory](https://cds.cern.ch/record/1456844) is a tool to construct probabilty distribution
functions from template histograms, constructing a likelihood function. In this exercise we will be
using HistFactory via [pyhf](https://pyhf.readthedocs.io), a python implementation of this tool. In addition,
we will be using the cabinetry package, which is a python library for constructing and
implementing HistFactory models.

The HistFactory template model can expressed like this:
![](../assets/img/hf_pdf.png){:width="50%"}

What's here?
* $$ \vec{n} $$ describes the observed channel data and $$ \vec{a} $$ describes the auxiliary data e.g. from calibration
* $$ \vec{a} $$ describes the unconstrained parameters (parameters of interest POI)and $$ \vec{\chi} $$ the constrained parameters (nuisance parameters NPs)
* $$ n_{cb} $$: the observed number of events, $$ \nu_{nb}(\eta \vert \chi) $$: expected number of events
* <span style="color:blue"> Main poisson p.d.f. for simultaneous measurement over multiple channels (or regions, like a signal region and a control region) and bins (over the histograms)</span>
* <span style="color:red">Constraint p.d.f which encodes systematic uncertainties: the actual function used depends on the parameter (e.g. it may be a Gaussian)</span>

This is a lot to take in but we'll press on regardless with implementation.

Essentially:

* `HistFactory` is a model used for binned statistical analysis
* `pyhf` is a python implementation of this model
* `cabinetry` creates a statistical model from a specification of cuts, systematics, samples, etc.
* `pyhf` then turns this model into a likelihood function

## Making models and fitting

>
> It should be noted that everything in this part is contained in the `use_cabinetry.py` script.
> This file can be found in your copy of the lesson repository
>
{: .testimonial}

Let's start by importing the necessary modules, most importantly `cabinetry`:
~~~
import logging
import cabinetry
~~~
{: .language-python}

### Systematics

In an analysis such as this one there are many systematics to consider, both experimental and theoretical (as encapsulated in the MC).
For the former these include trigger and selection efficiencies, jet energy scale and resolution,
b-tagging and misidentification efficiencies, and integrated luminosity. The latter can include uncertainties
due to choice of hadronizer, choice of generator, QCD scale choices, and the parton shower scale.
This isn't a complete list of systematics and here will only cover a few of these (more on this later).

### Histograms

Let's quickly inspect our histograms in the `histograms.root` file by running ROOT and opening
up a TBrowser.

>
> Try this if you have ROOT installed and have gone through the [ROOT pre-exercise](https://cms-opendata-workshop.github.io/workshop2022-lesson-cpp-root-python/).
> If not, skip this part and try it later. We'll show the output below anyway.
>
{: .testimonial}

~~~
root histograms.root
~~~
{: .language-bash}

~~~
   ------------------------------------------------------------------
  | Welcome to ROOT 6.26/00                        https://root.cern |
  | (c) 1995-2021, The ROOT Team; conception: R. Brun, F. Rademakers |
  | Built for linuxx8664gcc on Jun 14 2022, 14:46:00                 |
  | From tag , 3 March 2022                                          |
  | With c++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0                   |
  | Try '.help', '.demo', '.license', '.credits', '.quit'/'.q'       |
   ------------------------------------------------------------------

root [0] 
Attaching file histograms.root as _file0...
(TFile *) 0x563baef326f0
root [1] TBrowser b
(TBrowser &) Name: Browser Title: ROOT Object Browser
~~~
{: .output}

Click on one of the histogram titles on the left ofr `4j1b` to view it:

![](../assets/img/4j1b_ttbar.png){:width="50%"}

Click on another for `4j2b`:

![](../assets/img/4j2b_ttbar.png){:width="50%"}

Recall that our observable for the `4j1b` control region is the scalar sum of jet transverse momentum, $$ H_{T} $$ and our observable for the `4j2b` signal region is the mass of b-jet system $$ m_{b_{jj}} $$.

### Cabinetry workspace

A statistical model can be define in a declarative way using cabinetry, capturing the 
$$ \mathrm{region \otimes sample \otimes systematic} $$ structure. 

General settings `General:`, list of phase space regions such as signal and control regions `Regions:`, list of samples (MC and data) `Samples:`, list of systematic uncertainties `Systematics:`, and a list of normalization factors `NormFactors:`.

Let's have a look at each of the parts of the the configuration file:

#### General

~~~
General:
  Measurement: "CMS_ttbar"
  POI: "ttbar_norm"
  HistogramFolder: "histograms/"
  InputPath: "histograms.root:{RegionPath}_{SamplePath}{VariationPath}"
  VariationPath: ""
~~~
{: .language-yaml}

#### Regions

~~~
Regions:
  - Name: "4j1b CR"
    RegionPath: "4j1b"

  - Name: "4j2b SR"
    RegionPath: "4j2b"
~~~
{: .language-yaml}

#### Samples

~~~
Samples:
  - Name: "Pseudodata"
    SamplePath: "pseudodata"
    Data: True

  - Name: "ttbar"
    SamplePath: "ttbar"

  - Name: "W+jets"
    SamplePath: "wjets"

  - Name: "single top, t-channel"
    SamplePath: "single_top_t_chan"

  - Name: "single atop, t-channel"
    SamplePath: "single_atop_t_chan"
    
  - Name: "tW"
    SamplePath: "single_top_tW"
~~~ 
{: .language-yaml}

#### Systematics

~~~
Systematics:
  - Name: "ME variation"
    Type: "NormPlusShape"
    Up:
      VariationPath: "_ME_var"
    Down:
      Symmetrize: True
    Samples: "ttbar"

  - Name: "PS variation"
    Type: "NormPlusShape"
    Up:
      VariationPath: "_PS_var"
    Down:
      Symmetrize: True
    Samples: "ttbar"
~~~
{: .language-yaml}

Here we specify which systematics we want to take into account. In addition to the W+jets scale variations, b-tagging variations, and jet energy scale and resolution (shown in the full file) we show here for the `ttbar` samples `_ME_var` (what do the result look like if we choose another generator?) and `_PS_var` (what do the results look like if we use a different hadronizer?). 

#### NormFactors

~~~
NormFactors:
  - Name: "ttbar_norm"
    Samples: "ttbar"
    Nominal: 1.0
    Bounds: [0, 10]
~~~
{: .language-yaml}

### Running cabinetry and results

Let's load the `cabinetry` configuration file and combine the histograms into a `pyhf` workspace which we will save:
~~~
config = cabinetry.configuration.load("cabinetry_config.yml")
cabinetry.templates.collect(config)
ws = cabinetry.workspace.build(config)
cabinetry.workspace.save(ws, "workspace.json")
~~~
{: .language-python}

`pyhf` can be run on the command line to inspect the workspace:
~~~
pyhf inspect workspace | head -n 20
~~~
{: .language-bash}

which outputs the following:
~~~
                 Summary       
            ------------------  
               channels  2
                samples  5
             parameters  14
              modifiers  14

               channels  nbins
             ----------  -----
                4j1b CR   11  
                4j2b SR   11  

                samples
             ----------
                 W+jets
 single atop, t-channel
  single top, t-channel
                     tW
                  ttbar
~~~
{: .output}


Now we perform our maximum likelihood fit
~~~
model, data = cabinetry.model_utils.model_and_data(ws)
fit_results = cabinetry.fit.fit(model, data)
~~~
{: .language-python}

and visualize the pulls of parameters in the fit:
~~~

pull_fig = cabinetry.visualize.pulls(
    fit_results, exclude="ttbar_norm", close_figure=True, save_figure=True
)
~~~
{: .language-python}

![](../assets/img/pulls.png){:width="50%"}


Note that the figures produced by running the script or your commands are to be found in the
`figures/` directory.

~~~
poi_index = model.config.poi_index
print(f"\nfit result for ttbar_norm: {fit_results.bestfit[poi_index]:.3f} +/- {fit_results.uncertainty[poi_index]:.3f}")
~~~
{: .language-python}


What does the model look like before and after the fit? We can visualize each with the following
code:
~~~
model_prediction = cabinetry.model_utils.prediction(model)
figs = cabinetry.visualize.data_mc(model_prediction, data, close_figure=True)

model_prediction_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
figs = cabinetry.visualize.data_mc(model_prediction_postfit, data, close_figure=True)
~~~
{: .language-python}

We can see that there is very good post-fit agreement:
![](../assets/img/4j1b-CR_postfit.png){:width="50%"}
![](../assets/img/4j2b-SR_postfit.png){:width="50%"}

## References and further reading

* [cabinetry](https://cabinetry.readthedocs.io/en/latest/index.html)
* [pyhf](https://pyhf.readthedocs.io)
* [HistFactory](https://cds.cern.ch/record/1456844)

## Acknowledgements

Thanks to the authors of the [AGC using CMS data](https://github.com/iris-hep/analysis-grand-challenge) on which much of the episode was based.

{% include links.md %}
