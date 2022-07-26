---
title: "Coffea Analysis"
teaching: 20
exercises: 60
questions:
- "What is the general plan for the analysis?"
- "What are the signal and background datasets that we will need?"
objectives:
- "1st learning objective"
- "2nd learning objective"
keypoints:
- "keypoint 1"
- "keypoint 2"
---

## Datasets and pre-selection

As was mentioned in the previous episode, we will be working towards a measurement of the top and anti-top quark production cross section $$ \sigma_{t\bar{t}} $$.  The lepton+jets final state $$t\bar{t} \rightarrow (bW^{+})(\bar{b}W_{-}) \rightarrow bq\bar{q} bl^{-}\bar{\nu_{l}}$$ is characterized by one lepton (here we look at electrons and muons only), significant missing transverse energy, and four jets, two of which are identified as b-jets.

For this analysis we will the following datasets:

**FIX ME**


As you can see for yourself, several of these datasets are quite large (TB in size), therefore we need to skim them.  Also, as you were able to see previously, running CMSSW EDAnalyzers over these data (with the POET code, for instance) could be quite computing intensive.  One could also estimate the time it would take to run over all the datasets we need using a single computer. To be efficient, you will need a computer cluster, but we will leave that for the Cloud Computing lesson. Fortunately, we have prepared these skims already at CERN, using CERN/CMS computing infrastructure. The *skimmed* files we will be using were obtained in essentially the same POET way, except that we applied some *trigger* filtering and some *pre-selection* requirements.

We explicitly required:

* That the events *fired* at least one of these triggers: `HLT_Ele22_eta2p1_WPLoose_Gsf`, `HLT_IsoMu20_v`, `HLT_IsoTkMu20_v`.  We assume these triggers were unprescaled, but you know now, one should really check.
* That the event contains either at least one `tight` electron with $$p_{T} > 26$$ and $$\lvert\eta\rvert<2.1$$ or at least one `tight` muon with $$p_{T} > 26$$ and $$\lvert\eta\rvert<2.1$$.

These pre-selection filters reduce the output of the files significantly to the point that we a single file per each dataset is manageable.  A json file called `ntuples.json` was created in order to keep track of them and their metadata.  You can find this file in your copy of the `workshop2022-lesson-ttbarljetsanalysis-payload` repository. Ther is an additional file colled `ntuples_reduced.json`, which can be used if you are connecting from outside or decided not to download the datasets previously.

**FIXME**: show the json file here


## Building the basic analysis with Coffea

Don't forget to keep working on the python tools Docker container.  In order to advance, we need the improved schema on which we worked last section.  If you didn't manage to get it right, you can download it from [here](FIXME).  You can get it directly by doing

~~~
wget FIXME
~~~
{: .language-bash}

Let'start fresh, import the needed libraries and open an example file:

~~~
import uproot
import awkward as ak
import hist
from hist import Hist
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from agc_schema import AGCSchema
events = NanoEventsFactory.from_root('root://eospublic.cern.ch//eos/opendata/cms/upload/od-workshop/ws2021/myoutput_odws2022-ttbaljets-prodv2.0_merged.root', schemaclass=AGCSchema, treepath='events').events()
~~~
{: .language-python}

Let's make some histograms

First, start by applying a $$p_{T}$$ cut for the objects of interest, namely electrons, muons and jets.  To compare, first check the number of muons in each subarray of the original collection:

~~~
ak.num(events.electron, axis=1)
ak.num(events.muon, axis=-1)
ak.num(events.jet, axis=1)
~~~
{: .language-python}

Now, let's apply the $$p_{T}$$ and $$\eta$$ *mask* requirement:

~~~
# pT > 30 GeV for leptons & jets and eta requirements
selected_electrons = events.electron[(events.electron.pt > 30) & (abs(events.electron.eta)<2.1)]
selected_muons = events.muon[(events.muon.pt > 30) & (abs(events.muon.eta)<2.1)]
selected_jets = events.jet[(events.jet.corrpt > 30) & (abs(events.jet.eta)<2.4)]
~~~
{: .language-python}

See what we got:

~~~
ak.num(selected_electrons, axis=1)
ak.num(selected_muons, axis=1)
ak.num(selected_jets, axis=1)
~~~
{: .language-python}

Note that the number of events are still the same, for example:

~~~
ak.num(events.electron, axis=0)
ak.num(selected_electrons, axis=0)
~~~
{: .language-python}




~~~

~~~
{: .language-python}

~~~

~~~
{: .output}


{% include links.md %}
