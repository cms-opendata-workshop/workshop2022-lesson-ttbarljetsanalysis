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

A statistical model $$ f(\bf{x} \vert \bf{\phi}) $$ describes the probability of the data
$$ \bf{x} $$ given model paramters $$ \bf{\phi} $$.

[HistFactory](https://cds.cern.ch/record/1456844) is a tool to construct probabilty distribution
functions from template histograms, constructing a likelihood function. In this exercise we will
using HistFactory via [pyhf](https://pyhf.readthedocs.io), a python implementation of this tool. In addition, we will be using
the cabinetry package, which is a python library for constructing and implementing HistFactory models.

**NOTE: we should add install commands (using pip) for pyhf and cabinetry upstream in the episodes**

**Q: how much statistics do we want to cover here? POIs and NPs? Likelihood functions and NP constraints? etc.**

## Systematics

In an analysis such as this one there are many systematics to consider, both experimental and theoretical (as encapuslated in the MC). For the former these include trigger and selection efficiencies, jet energy scale and resolution, b-tagging and misidentification efficiencies, and integrated luminosity. The latter can include uncertainties due to choice of hadronizer, choice of generator, QCD scale choices, and the parton shower scale. This isn't a complete list of systematics and we will only cover X, Y, Z here.

## Variation histograms

## Cabinetry workspace

A statistical model can be define in a declarative way using cabinetry, capturing the 
$$ \mathrm{region \otimes sample \otimes systematic} $$ structure. 

**Show config file here**

General settings `General:`, list of phase space regions such as signal and control regions `Regions:`, list of samples (MC and data) `Samples:`, list of systematic uncertainties `Systematics:`, and a list of normalization factors `NormFactors:`.

### General

### Regions

### Samples

### Systematics

## NormFactors

{% include links.md %}
