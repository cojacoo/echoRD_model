import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import os, sys
import cPickle as pickle

#connect echoRD Tools
pathdir='../echoRD/' #path to echoRD
#pathdir='../' #path to echoRD
lib_path = os.path.abspath(pathdir)
sys.path.append(lib_path)
import vG_conv as vG
from hydro_tools import plotparticles_t,hydroprofile,plotparticles_specht

# Read Observations
obs=pd.read_csv('brspecht.dat',delimiter='\t')
obs.index=np.arange(-0.05,-1.,-0.1)


# Load echoRD setup
#connect to echoRD
import run_echoRD as rE
#connect and load project
[dr,mc,mcp,pdyn,cinf,vG]=rE.loadconnect(pathdir=pathdir,mcinif='mcini_specht4y')
#[mc,particles,npart,precTS]=mcp.echoRD_pick_out(mc,'weiherbach_testcase.pickle')
mcp.mcpick_out(mc,'specht4y_x2.pickle')
mc.advectref='Shipitalo'
[mc,particles,npart]=dr.particle_setup(mc)
precTS=pd.read_csv(mc.precf, sep=',',skiprows=3)

mc.prects=False
[thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
#[A,B]=plotparticles_t_obs(particles,obsx,thS/100.,mc,vG,store=True)

#shift sprinkling beginning
mc.advectref='Shipitalo'
mc.prects=False
precTS.tstart=60
precTS.tend=60+2.3*3600
precTS.total=0.02543
precTS.intense=precTS.total/(precTS.tend-precTS.tstart)
precTS

#update mc.a_velocity_real to Shipitalo 0.203 m/s
mc.a_velocity_real=-mc.a_velocity_real*0.203/mc.a_velocity_real[-1]
n_part_tot=precTS.intense.values*4680.*mc.mgrid.width.values/mc.particleA
tracer_part=mc.tracer_appl_Br*mc.particleD/n_part_tot

#NOISE KS
mu, sigma = 4.9e-6, 4.8e-6 # mean and standard deviation of observed ks at spechtacker
sl = np.random.normal(mu, 10**sigma, len(mc.soilgrid.ravel()))
ks_noise_factor=10**sl

# Run Model
t_end=3.*3600.
saveDT=20

#1: MDA
#2: MED
infiltmeth='MDA'
#3: RWdiff
#4: Ediss
exfiltmeth='Ediss'
#5: film_uconst
#6: dynamic u
film=True
#7: maccoat1
#8: maccoat10
#9: maccoat100
macscale=1. #scale the macropore coating 


runname='Weiherbach_noise_'

clogswitch=False
drained=pd.DataFrame(np.array([]))
leftover=0
output=60. #mind to set also in TXstore.index definition

dummy=np.floor(t_end/output)
#prepare store arrays
TSstore=np.zeros((int(dummy),np.shape(thS)[0],np.shape(thS)[1]))
NPstore=np.zeros((int(dummy),len(mc.zgrid[:,1])+1))
colnames=['part1','part2','part3','part4','part5']
TXstore=pd.DataFrame(np.zeros((int(dummy),len(colnames))),columns=colnames)
t=0.
#loop through plot cycles
for i in np.arange(dummy.astype(int)):
    #plot and store states
    plotparticles_specht(particles,mc,pdyn,vG,runname,t,i,saving=True,relative=False)
    [TXstore.iloc[i],NPstore[i,:]]=plotparticles_t(particles,thS/100.,mc,vG,runname,t,i,saving=True,store=True)
    #store thS
    TSstore[i,:,:]=thS
    
    [particles,npart,thS,leftover,drained,t]=rE.CAOSpy_rundx_noise(i*output,(i+1)*output,mc,pdyn,cinf,precTS,particles,leftover,drained,6.,splitfac=4,prec_2D=False,maccoat=macscale,saveDT=saveDT,clogswitch=clogswitch,infilt_method=infiltmeth,exfilt_method=exfiltmeth,film=film,dynamic_pedo=True,ksnoise=ks_noise_factor)

    if i/10.==np.round(i/10.):
        f = open(''.join(['./results/N',runname,'TS.pick']),"wb")
        pickle.dump(pickle.dumps([pickle.dumps(TSstore),pickle.dumps(TXstore),pickle.dumps(NPstore)]), f, protocol=2)
        f.close()

        f = open(''.join(['./results/N',runname,'particles.pick']),"wb")
        pickle.dump(pickle.dumps(pickle.dumps(particles)), f, protocol=2)
        f.close()

