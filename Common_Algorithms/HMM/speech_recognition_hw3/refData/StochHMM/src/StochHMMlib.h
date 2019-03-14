//
//  StochHMMlib.h
//  StochHMM
//
//  Created by Paul Lott on 2/20/13.
//  Copyright (c) 2013 Korf Lab, Genome Center, UC Davis, Davis, CA. All rights reserved.
//

#ifndef StochHMM_StochHMMlib_h
#define StochHMM_StochHMMlib_h

#include "sequences.h"
#include "sequence.h"
#include "seqTracks.h"
#include "externDefinitions.h"
#include "options.h"
#include "weight.h"
#include "transitions.h"
#include "modelTemplate.h"
#include "externalFuncs.h"
#include "emm.h"
#include "track.h"
#include "lexicalTable.h"
#include "state.h"
#include "hmm.h"
#include "userFunctions.h"
#include "text.h"
#include "stochTypes.h"
#include "stochMath.h"
#include "index.h"
#include "PDF.h"
#include "pwm.h"
#include "trellis.h"
#include "stochTable.h"
#include "traceback_path.h"

#define VERSION 0.37

/*!
   @mainpage  StochHMM: A Flexible C++ Hidden Markov Model Library and Application
	
 StochHMM is a free, open source C++ Library and application that implements HMM
 from simple text files.   It implements traditional HMM algorithms in addition it
 providing additional flexibility.  The additional flexibility is achieved by
 allowing researchers to integrate additional data sources and application into
 the HMM framework.
 
 For documentation on model syntax and designing a model, see Github wiki.
 
 ###http://www.github.com/KorfLab/StochHMM/wiki
 
 ##Integrating Data
 Here are a few of the ways that StochHMM allows the users to integrate additional
 data sources:
	1. Multiple Emission States
	2. Weighting or Explicitly Defining State paths on a sequence
	3. Linking States Emissions/Transitions to external user-defined functions
 
 
 ##Multiple Emission States
 
 StochHMM allows the user to provide multiple sequences.   These sequences are then
 handled by the emissions.  These sequences can be REAL numbers or discrete characters/words.
 StochHMM allows each state to have many emissions (Discrete or Continuous).  Discrete emissions
 can be independent of each other or joint distributions.   The continuous emissions
 can be considered in multiple ways.   1)  They can be considered as raw probabilities which 
 will be integrated without transformation.  2) They can be considered as values to be plugged
 into a Univariate Probability Distribution Function or Multivariate PDF (In the case of multiple
 REAL sequences.
 
 Each states emissions are user-defined, so one state may have emissions from two
 different sequences, while another may only have a single emission from a single sequence.
 
 ##Weighting or Explicitly Defining State paths to follow on a sequence.
 
 Often, we have some prior knowledge about the sequence.   If this is the case,
 we may want to integrate that into the model, without redesigning or retraining the model (a timely endeavor).
 StochHMM allows the user to explicitly define a State path (By name of state, or category of state).
 In addition, StochHMM also allows the user to weight a states path (By name of State or category of state defined by user)
 This allows the user to restrict the predicted path or weight their prior knowledge.
 
 
 ##Linking States Emissions or Transitions to external user-defined functions
 
 When that transition/emission is evaluated the function is called and can provide an emission. While 
 this may provide one way of addressing a weakness of HMMs, which is that they do not handle long
 range dependencies.  We see it rather as a way to link together existing utilities or functions that 
 provide additional information to the decoding algorithms.   In this way, we can link divergent 
 datasets or functions within the HMM trellis in order to arrive at a better prediction. 

 
 ##Brief list of features implemented in StochHMM:
	- General settings within Hidden Markov Models
		1. User-defined HMM model via simple human readable text file
		2. User-defined Alphabet
		3. User-defined Ambiguous Characters
	- States
		1. Emissions
			- Multiple emission states (Discrete / Continuous)
				- Independent (Single or Multiple Discrete)
				- Joint Distribution (Multiple Discrete)
				- Univariate PDF (Single Sequence -  Continuous)
				- Multivariate PDF (Multiple Sequence - Continuous)
			- Linkable to user-defined function
		2. Transitions
			- Standard Transitions
			- Lexical Transitions (Single or multiple emission)
			- Explicit Duration Transitions
			- Linkable to user-defined functions
	- Decoding
		1. Traditional Decoding Algorithms
			- Forward/Backward/Posterior
			- Viterbi
			- N-best Viterbi
		2. Stochastic Sampling Decoding Algorithms
			- Stochastic Forward
			- Stochastic Viterbi
			- Stochastic Posterior
	- Decoding Traceback Path output formats
		- State Path Index
		- State Path Label
		- GFF
		- Hit Table (Stochastic Algorithms)
 
  
 ##Developed by:
 
 ##[Korf Lab](http://korflab.ucdavis.edu/)
 
 Korf Lab, Genome Center, University of California, Davis
	- [Ian Korf](http://korflab.ucdavis.edu/bio_ian.html)
	- [Paul Lott](http://korflab.ucdavis.edu/bio_paul.html)
	- [Keith Dunaway](http://korflab.ucdavis.edu/bio_keith_d.html)
	- [Ken Yu](http://korflab.ucdavis.edu/bio_ken.html)
 
 ##For suggestions or support:
	- korflab AT ucdavis DOT edu
	- [KorfLab Github](https://github.com/KorfLab/StochHMM)
	- Google Groups
		- [StochHMM-dev](https://groups.google.com/forum/?fromgroups#!forum/stochhmm-dev)
		- [StochHMM-Forum](https://groups.google.com/forum/?fromgroups#!forum/stochhmm-forum)
 
 
 ##References:
 
 1. Schroeder, D.I., Blair J.D., Lott P., Yu H.O., Hong D., Crary F., Ashwood P., Walker C.
 , Korf I., Robinson W.P., LaSalle J.M.. The human placenta methylome. PNAS 15:6037-6042 (2013)
 
 2.  Schroeder, D. I., Lott, P., Korf, I. & LaSalle, J. M. Large-scale
 methylation domains mark a functional subset of neuronally expressed
 genes. Genome Res 21, 1583–1591 (2011).
 
 3.	Ginno, P. A., Lott, P. L., Christensen, H. C., Korf, I. & Chédin,
 F. R-loop formation is a distinctive characteristic of unmethylated
 human CpG island promoters. Mol. Cell 45, 814–825 (2012).
 
 
 
 ## Open Source Licensing:
 
 The MIT License (MIT)
 
 Copyright (c) 2007-2012 Paul Lott, Ian Korf, Korf Lab, Genome Center, UC Davis, Davis, CA. All rights reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 
 */

#endif
