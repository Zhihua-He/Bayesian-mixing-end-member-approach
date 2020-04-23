data {
//size of sample observations
int<lower=1> I;
int<lower=1> S;
int<lower=1> J;
int<lower=1> J_;
int<lower=1> J1;
int<lower=1> J1_;
int<lower=1> J2;
int<lower=1> J2_;
int<lower=1> J3;
int<lower=1> J3_;
int<lower=1> J4;
int<lower=1> J4_;
int<lower=1> K;
int<lower=1> K_;
int<lower=1> K1;
int<lower=1> K1_;
int<lower=1> K2;
int<lower=1> K2_;
int<lower=1> K3;
int<lower=1> K3_;
int<lower=1> K4;
int<lower=1> K4_;
//isotope measurements
vector[I] Tacs1[J1];     //all tracer measurements for groundwater
vector[I] Tacs2[J2];     //all tracer measurements for rainfall
vector[I] Tacs3[J3];     //all tracer measurements for snowmelt
vector[I] Tacs4[J4];     //all tracer measurements for glacier melt
vector[I] Tacr[J];       //all tracer measurements for streamflow
vector[I] Tacr_[J_];     //spatial average tracers for streamflow
vector[I] Tacs1_[J1_];   //spatial average tracers for groundwater
vector[I] Tacs2_[J2_];   //spatial average tracers for rainfall
vector[I] Tacs3_[J3_];   //spatial average tracers for snowmelt
vector[I] Tacs4_[J4_];   //spatial average tracers for glacier melt
//Electrical conductivity measurements
vector[K] ECr;           //all EC measurements for streamflow
vector[K1] ECs1;         //all EC measurements for groundwater
vector[K2] ECs2;         //all EC measurements for rainfall
vector[K3] ECs3;         //all EC measurements for snowmelt
vector[K4] ECs4;         //all EC measurements for glacier melt
vector[K_] ECr_;         //spatial average EC measurements for streamflow
vector[K1_] ECs1_;       //spatial average EC measurements for groundwater
vector[K2_] ECs2_;       //spatial average EC measurements for rainfall
vector[K3_] ECs3_;       //spatial average EC measurements for snowmelt
vector[K4_] ECs4_;       //spatial average EC measurements for glacier melt

}
parameters {
vector[I] mus[S];          //means of water isotopes
cov_matrix[I] Tau[S+1];    //convariance matrix of water isotopes
simplex[S] fi;             //contributions of water sources to the streamflow
real<lower=0> muec[S];     //means of EC
real<lower=0> decta[S+1];  //variance of EC
vector[S+1] mu[I];         //means of spatial water isotopes
vector<lower=0>[S+1] deta[I]; // variance of spatial  water isotopes
real<lower=0> muecs[S+1];  //means of spatial EC
real<lower=0> dect[S+1];   // variance of spatial  EC
cov_matrix[I] namu;      //pre-defined parameter for the wishart distribution
vector<lower=0,upper=10>[I] ba[S];       //means of ρ
vector<lower=0,upper=10>[I] bc[S];     //means of ψ
vector<lower=0,upper=5>[I] frac[S];          //fractionation
vector<lower=0,upper=5>[I] de[S];          //fractionation variance
}
transformed parameters{
//mixing of water sources to the streamflow
vector[S] alpha;           //alpha for fi
vector[I] mur;
real muecr;
vector[I] fmu[S];          //fractionation variance

   for (s in 1:S) {
	   alpha[s]=ba[s,1]+ba[s,2];
   }
	for (i in 1:I)  {
	 mur[i] = fi[1]'*(mus[1,i]+frac[1,i])+fi[2]'*(mus[2,i]+frac[2,i])+fi[3]'*(mus[3,i]+frac[3,i])+fi[4]'*(mus[4,i]+frac[4,i]);
	}
	 muecr=fi[1]'*muec[1]+fi[2]'*muec[2]+fi[3]'*muec[3]+fi[4]'*muec[4];
  for (s in 1:S) {
     for (i in 1:I) {	 	  
  	  fmu[s,i]=mus[s,i]+frac[s,i];
 		}
 	}

}
model {
	fi~ dirichlet(alpha); // prior for contributions of water sources
	for (s in 1:S){
		ba[s,] ~ multi_normal(bc[s,],Tau[s]);
	}
	
//likelihood for the spatial water tracers	
	for (i in 1:I) {
		target += normal_lpdf(Tacs1_[,i]|mu[i,1],deta[i,1]);     
		target += normal_lpdf(Tacs2_[,i]|mu[i,2],deta[i,2]);
		target += normal_lpdf(Tacs3_[,i]|mu[i,3],deta[i,3]);
		target += normal_lpdf(Tacs4_[,i]|mu[i,4],deta[i,4]);	
		target += normal_lpdf(Tacr_[,i]|mu[i,S+1],deta[i,S+1]);
		}
		target += normal_lpdf(ECs1_|muecs[1],dect[1]);
		target += normal_lpdf(ECs2_|muecs[2],dect[2]);
		target += normal_lpdf(ECs3_|muecs[3],dect[3]);
		target += normal_lpdf(ECs4_|muecs[4],dect[4]);	
		target += normal_lpdf(ECr_|muecs[S+1],dect[S+1]);

//prior distributions for mean and variance of water tracers		
	for (i in 1:I) {
		for (s in 1:S) {
		mus[s,i] ~ normal(mu[i,s],deta[i,s]);
		Tau[s]~ wishart(2,namu);
	}
	target += normal_lpdf(mur[i]|mu[i,S+1],deta[i,S+1]);
	}
	for (s in 1:S) {
	muec[s]~ normal(muecs[s],dect[s]);
	frac[s,] ~ multi_normal(de[s,],Tau[s]);
	}
	
	Tau[S+1]~ wishart(2,namu);
	target += normal_lpdf(muecr|muecs[S+1],dect[S+1]);
//likelihood for the whole water tracer
		target += normal_lpdf(ECs1|muec[1],decta[1]);
		target += normal_lpdf(ECs2|muec[2],decta[2]);
		target += normal_lpdf(ECs3|muec[3],decta[3]);
		target += normal_lpdf(ECs3|muec[4],decta[4]);
		
		target += multi_normal_lpdf(Tacs1|mus[1,], Tau[1]);	
		target += multi_normal_lpdf(Tacs2|mus[2,], Tau[2]);
		target += multi_normal_lpdf(Tacs3|mus[3,], Tau[3]);		
		target += multi_normal_lpdf(Tacs3|mus[4,], Tau[4]);	

	target += multi_normal_lpdf(Tacr|mur, Tau[S+1]);
	target += normal_lpdf(ECr|muecr,decta[S+1]);
}
//generated quantities {
//vector[I] Tacr_c;
//real ECr_c;
//Tacr_c=lognormal_rng(mur,Taur);
//ECr_c = lognormal_rng(muec[1],decta[1]);
//}