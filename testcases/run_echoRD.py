import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.constants as const
import os, sys

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def loadconnect(pathdir='./', mcinif='mcini', oldvers=False):
    lib_path = os.path.abspath(pathdir)
    sys.path.append(lib_path)

    import dataread as dr
    if oldvers:
        import mcpickle as mcp
    else:
        import mcpickle2 as mcp
    import infilt as cinf
    import partdyn_d2 as pdyn
    mc = __import__(mcinif)
    import vG_conv as vG

    return(dr,mc,mcp,pdyn,cinf,vG)

def preproc_echoRD(mc, dr, mcp, pickfile='test.pickle'):
    mc=dr.dataread_caos(mc)
    mcp.mcpick_in(mc,pickfile)
    return mc

def pickup_echoRD(mc, mcp, dr, pickfile='test.pickle'):
    mcp.mcpick_out(mc,pickfile)
    [mc,particles,npart]=dr.particle_setup(mc)
    precTS=pd.read_csv(mc.precf, sep=',',skiprows=3)

    return(mc,particles,npart,precTS)

def particle_setup_obs(theta_obs,mc,vG,dr,pdyn):
    moistdomain=np.tile(theta_obs,int(mc.mgrid.latgrid)).reshape((mc.mgrid.latgrid,mc.mgrid.vertgrid)).T
        
    # define particle size
    # WARNING: as in any model so far, we have a volume problem here. 
    #          we consider all parts of the domain as static in volume at this stage. 
    #          however, we will work on a revision of this soon.
    mc.gridcellA=mc.mgrid.vertfac*mc.mgrid.latfac
    mc.particleA=abs(mc.gridcellA.values)/(2*mc.part_sizefac) #assume average ks at about 0.5 as reference of particle size
    mc.particleD=2.*np.sqrt(mc.particleA)/np.pi
    mc.particleV=3./4.*np.pi*(mc.particleD/2.)**3.
    mc.particlemass=dr.waterdensity(np.array(20),np.array(-9999))*mc.particleV #assume 20C as reference for particle mass
                                                                            #DEBUG: a) we assume 2D=3D; b) change 20C to annual mean T?

    # define macropore capacity based on particle size
    # we introduce a scale factor for converting macropore space and particle size
    mc.maccap=np.round(mc.md_area/((mc.particleD**2)*np.pi*mc.macscalefac)).astype(int)

    # convert theta to particles
    # npart=moistdomain*(2*mc.part_sizefac)
    npart=np.floor(mc.part_sizefac*vG.thst_theta(moistdomain,mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid)), mc.soilmatrix.tr[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid)))).astype(int)

    # setup particle domain
    particles=pd.DataFrame(np.zeros(int(np.sum(npart))*8).reshape(int(np.sum(npart)),8),columns=['lat', 'z', 'conc', 'temp', 'age', 'flag', 'fastlane', 'advect'])
    particles['cell']=pd.Series(np.zeros(int(np.sum(npart)),dtype=int), index=particles.index)
    # distribute particles
    k=0
    npartr=npart.ravel()
    cells=len(npartr)

    for i in np.arange(cells):
        j=int(npartr[i])
        particles.cell[k:(k+j)]=i
        rw,cl=np.unravel_index(i,(mc.mgrid.vertgrid,mc.mgrid.latgrid))
        particles.lat[k:(k+j)]=(cl+np.random.rand(j))*mc.mgrid.latfac.values
        particles.z[k:(k+j)]=(rw+np.random.rand(j))*mc.mgrid.vertfac.values
        k+=j

    particles.fastlane=np.random.randint(len(mc.t_cdf_fast.T), size=len(particles))
    particles.advect=pdyn.assignadvect(int(np.sum(npart)),mc,particles.fastlane.values,True)

    mc.mgrid['cells']=cells
    return [mc,particles.iloc[0:k,:],npart]


def start_echoRDdx(mc,particles,npart,precTS,pdyn,cinf,runname='echoRD',t_end=3600.,output=60.,start_offset=0.,splitfac=10):
    [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
    drained=pd.DataFrame(np.array([]))
    leftover=0
    plotparticles_t(runname,0.,0,particles,(thS/100.).reshape(np.shape(npart)),mc)
    #loop through plot cycles
    dummy=np.floor(t_end/output)
    for i in np.arange(dummy.astype(int)):
        [particles,npart,thS,leftover,drained,t]=CAOSpy_rundx(i*output+start_offset,(i+1)*output+start_offset,mc,pdyn,cinf,precTS,particles,leftover,drained,splitfac=splitfac)
        plotparticles_t(runname,t,i+1,particles,(thS/100.).reshape(np.shape(npart)),mc)

def start_echoRDxstore(mc,particles,npart,precTS,pdyn,cinf,runname='echoRD',t_end=3600.,output=60.,start_offset=0.,splitfac=10,maccoat=10.,exfilt_method='Ediss'):
    [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
    drained=pd.DataFrame(np.array([]))
    leftover=0
    plotparticles_t(runname,0.,0,particles,(thS/100.).reshape(np.shape(npart)),mc)
    #loop through plot cycles
    dummy=np.floor(t_end/output)
    TSstore=np.zeros((int(dummy),np.shape(thS)[0],np.shape(thS)[1]))
    for i in np.arange(dummy.astype(int)):
        [particles,npart,thS,leftover,drained,t]=CAOSpy_rundx(i*output+start_offset,(i+1)*output+start_offset,mc,pdyn,cinf,precTS,particles,leftover,drained,splitfac=splitfac,prec_2D=True,maccoat=maccoat,exfilt_method=exfilt_method)
        plotparticles_t(runname,t,i+1,particles,(thS/100.).reshape(np.shape(npart)),mc)
        TSstore[i,:,:]=thS
    return TSstore


def CAOSpy_rundx(tstart,tstop,mc,pdyn,cinf,precTS,particles,leftover,drained,dt_max=1.,splitfac=10,prec_2D=False,maccoat=10.,exfilt_method='Ediss',saveDT=True,vertcalfac=1.,latcalfac=1.,clogswitch=False,infilt_method='MDA',film=True,infiltscale=False):
    if run_from_ipython():
        from IPython import display

    timenow=tstart
    prec_part=0. #precipitation which is less than one particle to accumulate
    acc_mxinf=0. #matrix infiltration may become very small - this shall handle that some particles accumulate to infiltrate
    exfilt_p=0. #exfiltration from the macropores
    s_red=0.
    #loop through time
    while timenow < tstop:
        [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
        if saveDT==True:
            #define dt as Courant/Neumann criterion
            dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.nanmax(mc.D[np.amax(thS),:]))
            dt_ku=-mc.mgrid.vertfac.values[0]/np.nanmax(mc.ku[np.amax(thS),:])
            dt=np.amin([dt_D,dt_ku,dt_max,tstop-timenow])
        else:
            if type(saveDT)==float:
                #define dt as pre-defined
                dt=np.amin([saveDT,tstop-timenow])
            elif type(saveDT)==int:
                #define dt as modified  Corant/Neumann criterion
                dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.nanmax(mc.D[np.amax(thS),:]))*saveDT
                dt_ku=-mc.mgrid.vertfac.values[0]/np.nanmax(mc.ku[np.amax(thS),:])*saveDT
                dt=np.amin([dt_D,dt_ku,dt_max,tstop-timenow])
        #INFILTRATION
        [p_inf,prec_part,acc_mxinf]=cinf.pmx_infilt(timenow,precTS,prec_part,acc_mxinf,thS,mc,pdyn,dt,0.,prec_2D,particles.index[-1],infilt_method,infiltscale) #drain all ponding // leftover <-> 0.
        particles=pd.concat([particles,p_inf])
        
        #DIFFUSION
        [particles,thS,npart,phi_mx]=pdyn.part_diffusion_split(particles,npart,thS,mc,dt,False,splitfac,vertcalfac,latcalfac)
        #ADVECTION
        if not particles.loc[(particles.flag>0) & (particles.flag<len(mc.maccols)+1)].empty:
            [particles,s_red,exfilt_p]=pdyn.mac_advection(particles,mc,thS,dt,clogswitch,maccoat,exfilt_method,film=film)
        #INTERACT
        particles=pdyn.mx_mp_interact_nobulk(particles,npart,thS,mc,dt)

        if run_from_ipython():
            display.clear_output()
            display.display_pretty(''.join(['time: ',str(timenow),'s  |  precip: ',str(len(p_inf)),' particles  |  mean v(adv): ',str(particles.loc[particles.flag>0,'advect'].mean()),' m/s  |  exfilt: ',str(int(exfilt_p)),' particles']))
        else:
            print 'time: ',timenow,'s'

        #CLEAN UP DATAFRAME
        drained=drained.append(particles[particles.flag==len(mc.maccols)+1])
        particles=particles[particles.flag!=len(mc.maccols)+1]
        pondparts=(particles.z<0.)
        leftover=np.count_nonzero(-pondparts)
        particles.cell[particles.cell<0]=mc.mgrid.cells.values
        particles=particles[pondparts]
        timenow=timenow+dt

    return(particles,npart,thS,leftover,drained,timenow)

def CAOSpy_rundx_noise(tstart,tstop,mc,pdyn,cinf,precTS,particles,leftover,drained,dt_max=1.,splitfac=10,prec_2D=False,maccoat=10.,exfilt_method='Ediss',saveDT=True,vertcalfac=1.,latcalfac=1.,clogswitch=False,infilt_method='MDA',film=True,dynamic_pedo=True,ksnoise=1.):
    if run_from_ipython():
        from IPython import display

    timenow=tstart
    prec_part=0. #precipitation which is less than one particle to accumulate
    acc_mxinf=0. #matrix infiltration may become very small - this shall handle that some particles accumulate to infiltrate
    exfilt_p=0. #exfiltration from the macropores
    s_red=0.
    #loop through time
    while timenow < tstop:
        [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
        if saveDT==True:
            #define dt as Courant/Neumann criterion
            dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.nanmax(mc.D[np.amax(thS),:]))
            dt_ku=-mc.mgrid.vertfac.values[0]/np.nanmax(mc.ku[np.amax(thS),:])
            dt=np.amin([dt_D,dt_ku,dt_max,tstop-timenow])
        else:
            if type(saveDT)==float:
                #define dt as pre-defined
                dt=np.amin([saveDT,tstop-timenow])
            elif type(saveDT)==int:
                #define dt as modified  Corant/Neumann criterion
                dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.nanmax(mc.D[np.amax(thS),:]))*saveDT
                dt_ku=-mc.mgrid.vertfac.values[0]/np.nanmax(mc.ku[np.amax(thS),:])*saveDT
                dt=np.amin([dt_D,dt_ku,dt_max,tstop-timenow])
        #INFILTRATION
        [p_inf,prec_part,acc_mxinf]=cinf.pmx_infilt(timenow,precTS,prec_part,acc_mxinf,thS,mc,pdyn,dt,0.,prec_2D,particles.index[-1],infilt_method) #drain all ponding // leftover <-> 0.
        particles=pd.concat([particles,p_inf])
        
        #DIFFUSION
        [particles,thS,npart,phi_mx]=pdyn.part_diffusion_split(particles,npart,thS,mc,dt,False,splitfac,vertcalfac,latcalfac,dynamic_pedo=True,ksnoise=ksnoise)
        #ADVECTION
        if not particles.loc[(particles.flag>0) & (particles.flag<len(mc.maccols)+1)].empty:
            [particles,s_red,exfilt_p]=pdyn.mac_advection(particles,mc,thS,dt,clogswitch,maccoat,exfilt_method,film=film,dynamic_pedo=True,ksnoise=ksnoise)
        #INTERACT
        particles=pdyn.mx_mp_interact_nobulk(particles,npart,thS,mc,dt,dynamic_pedo=True,ksnoise=ksnoise)

        if run_from_ipython():
            display.clear_output()
            display.display_pretty(''.join(['time: ',str(timenow),'s  |  precip: ',str(len(p_inf)),' particles  |  mean v(adv): ',str(particles.loc[particles.flag>0,'advect'].mean()),' m/s  |  exfilt: ',str(int(exfilt_p)),' particles']))
        else:
            print 'time: ',timenow,'s'

        #CLEAN UP DATAFRAME
        drained=drained.append(particles[particles.flag==len(mc.maccols)+1])
        particles=particles[particles.flag!=len(mc.maccols)+1]
        pondparts=(particles.z<0.)
        leftover=np.count_nonzero(-pondparts)
        particles.cell[particles.cell<0]=mc.mgrid.cells.values
        particles=particles[pondparts]
        timenow=timenow+dt

    return(particles,npart,thS,leftover,drained,timenow)

def CAOSpy_rund_diffonly(tstart,tstop,mc,pdyn,cinf,precTS,particles,leftover,drained,dt_max=1.,splitfac=10,prec_2D=False,saveDT=True,vertcalfac=1.,latcalfac=1.):
    if run_from_ipython():
        from IPython import display

    timenow=tstart
    prec_part=0. #precipitation which is less than one particle to accumulate
    acc_mxinf=0. #matrix infiltration may become very small - this shall handle that some particles accumulate to infiltrate
    exfilt_p=0. #exfiltration from the macropores
    s_red=0.
   #loop through time
    while timenow < tstop:
        [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
        if saveDT==True:
            #define dt as Courant/Neumann criterion
            dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.nanmax(mc.D[np.amax(thS),:]))
            dt_ku=-mc.mgrid.vertfac.values[0]/np.nanmax(mc.ku[np.amax(thS),:])
            dt=np.amin([dt_D,dt_ku,dt_max,tstop-timenow])
        else:
            if type(saveDT)==float:
                #define dt as pre-defined
                dt=np.amin([saveDT,tstop-timenow])
            elif type(saveDT)==int:
                #define dt as modified  Corant/Neumann criterion
                dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.nanmax(mc.D[np.amax(thS),:]))*saveDT
                dt_ku=-mc.mgrid.vertfac.values[0]/np.nanmax(mc.ku[np.amax(thS),:])*saveDT
                dt=np.amin([dt_D,dt_ku,tstop-timenow])
        #INFILTRATION
        [p_inf,prec_part,acc_mxinf]=cinf.pmx_infilt(timenow,precTS,prec_part,acc_mxinf,thS,mc,pdyn,dt,0.,prec_2D,particles.index[-1]) #drain all ponding // leftover <-> 0.
        p_inf.flag=0
        particles=pd.concat([particles,p_inf])
        
        #DIFFUSION
        [particles,thS,npart,phi_mx]=pdyn.part_diffusion_split(particles,npart,thS,mc,dt,False,splitfac,vertcalfac,latcalfac)
        
        if run_from_ipython():
            display.clear_output()
            display.display_pretty(''.join(['time: ',str(timenow),'s  |  precip: ',str(len(p_inf)),' particles']))
        else:
            print 'time: ',timenow,'s'

        #CLEAN UP DATAFRAME
        drained=drained.append(particles[particles.flag==len(mc.maccols)+1])
        particles=particles[particles.flag!=len(mc.maccols)+1]
        pondparts=(particles.z<0.)
        leftover=np.count_nonzero(-pondparts)
        particles.cell[particles.cell<0]=mc.mgrid.cells.values
        particles=particles[pondparts]
        timenow=timenow+dt

    return(particles,npart,thS,leftover,drained,timenow)


def plotparticles2(runname,t,ix,particles,npart,mc):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec
    
    fig=plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,2], height_ratios=[1,5])
    ax1 = plt.subplot(gs[0])
    ax11 = ax1.twinx()
    advect_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age>0.)),'lat'].values).astype(np.int))
    old_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age<=0.)),'lat'].values).astype(np.int))
    ax1.plot((np.arange(0,len(advect_dummy))/100.)[1:],advect_dummy[1:],'b-')
    ax11.plot((np.arange(0,len(old_dummy))/100.)[1:],old_dummy[1:],'g-')
    ax11.set_ylabel('Particle Count', color='g')
    ax11.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_ylabel('New Particle Count', color='b')
    ax1.set_xlabel('Lat [m]')
    ax1.set_title('Lateral Particles Concentration')
    
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    ax2.text(0.1, 0.8, 'Particles @ t='+str(np.round(t/60.))+'min', fontsize=20)
    
    ax3 = plt.subplot(gs[2])
    plt.imshow(sp.ndimage.filters.median_filter(npart,size=mc.smooth),vmin=1, vmax=mc.part_sizefac, cmap='jet')
    #plt.imshow(npart)
    plt.colorbar()
    plt.xlabel(''.join(['Width [cells a ',str(np.round(1000*mc.mgrid.latfac.values[0],decimals=1)),' mm]']))
    plt.ylabel(''.join(['Depth [cells a ',str(np.round(1000*mc.mgrid.vertfac.values[0],decimals=1)),' mm]']))
    plt.title('Particle Density')
    plt.tight_layout()

    ax4 = plt.subplot(gs[3])
    #ax41 = ax4.twiny()
    z1=np.append(particles.loc[((particles.age>0.)),'z'].values,mc.onepartpercell[1][:mc.mgrid.vertgrid.values.astype(int)])
    advect_dummy=np.bincount(np.round(-100.0*z1).astype(np.int))-1
    old_dummy=np.bincount(np.round(-100.0*particles.loc[((particles.age<=0.)),'z'].values).astype(np.int))
    ax4.plot(advect_dummy,(np.arange(0,len(advect_dummy))/-100.),'r-',label='new particles')
    ax4.plot(advect_dummy+old_dummy,(np.arange(0,len(old_dummy))/-100.),'b-',label='all particles')
    ax4.plot(old_dummy,(np.arange(0,len(old_dummy))/-100.),'g-',label='old particles')
    ax4.set_xlabel('Particle Count')
    #ax4.set_xlabel('New Particle Count', color='r')
    ax4.set_ylabel('Depth [m]')
    #ax4.set_title('Number of Particles')
    ax4.set_ylim([mc.mgrid.depth.values,0.])
    ax4.set_xlim([0.,np.max(old_dummy+advect_dummy)])
    #ax41.set_xlim([0.,np.max(old_dummy[1:])])
    #ax41.set_ylim([mc.mgrid.depth.values,0.])
    handles1, labels1 = ax4.get_legend_handles_labels() 
    #handles2, labels2 = ax41.get_legend_handles_labels() 
    ax4.legend(handles1, labels1, loc=4)
    #    ax41.legend(loc=4)
    plt.savefig(''.join(['./results/',runname,str(ix).zfill(3),'.png']))
    #savefig('runname %(i)03d .png'.translate(None, ' '))
    plt.close(fig)

def plotparticles_t(runname,t,ix,particles,thS,mc):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec
    
    fig=plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2,1], height_ratios=[1,5])
    ax1 = plt.subplot(gs[0])
    ax11 = ax1.twinx()
    advect_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age>0.)),'lat'].values).astype(np.int))
    old_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age<=0.)),'lat'].values).astype(np.int))
    ax1.plot((np.arange(0,len(advect_dummy))/100.)[1:],advect_dummy[1:],'b-')
    ax11.plot((np.arange(0,len(old_dummy))/100.)[1:],old_dummy[1:],'g-')
    ax11.set_ylabel('Particle Count', color='g')
    ax11.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_ylabel('New Particle Count', color='b')
    ax1.set_xlabel('Lat [m]')
    ax1.set_title('Lateral Particles Concentration')
    
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    ax2.text(0.1, 0.8, 't='+str(np.round(t/60.,1))+'min', fontsize=20)
    
    ax3 = plt.subplot(gs[2])
    plt.imshow(sp.ndimage.filters.median_filter(thS,size=mc.smooth),vmin=0., vmax=1., cmap='Blues')
    #plt.imshow(npart)
    plt.colorbar()
    plt.xlabel('Width [cells a 5 mm]')
    plt.ylabel('Depth [cells a 5 mm]')
    plt.title('Particle Density')
    plt.tight_layout()

    ax4 = plt.subplot(gs[3])
    #ax41 = ax4.twiny()
    onez=np.arange(0.,mc.mgrid.depth-0.004,-0.01)-0.004
    z1=np.append(particles.loc[((particles.age>0.)),'z'].values,onez)
    advect_dummy=np.bincount(np.round(-100.0*z1).astype(np.int))-1
    z2=np.append(particles.loc[((particles.age<=0.)),'z'].values,onez)
    old_dummy=np.bincount(np.round(-100.0*z2).astype(np.int))-1
    ax4.plot(advect_dummy,(np.arange(0,len(advect_dummy))/-100.),'r-',label='new particles')
    ax4.plot(advect_dummy+old_dummy,(np.arange(0,len(old_dummy))/-100.),'b-',label='all particles')
    ax4.plot(old_dummy,(np.arange(0,len(old_dummy))/-100.),'g-',label='old particles')
    ax4.set_xlabel('Particle Count')
    #ax4.set_xlabel('New Particle Count', color='r')
    ax4.set_ylabel('Depth [m]')
    #ax4.set_title('Number of Particles')
    ax4.set_ylim([mc.mgrid.depth.values,0.])
    ax4.set_xlim([0.,np.max(old_dummy+advect_dummy)])
    #ax41.set_xlim([0.,np.max(old_dummy[1:])])
    #ax41.set_ylim([mc.mgrid.depth.values,0.])
    handles1, labels1 = ax4.get_legend_handles_labels() 
    #handles2, labels2 = ax41.get_legend_handles_labels() 
    ax4.legend(handles1, labels1, loc=4)
    #    ax41.legend(loc=4)
    plt.savefig(''.join(['./results/',runname,'t_',str(ix).zfill(3),'.png']))
    #savefig('runname %(i)03d .png'.translate(None, ' '))
    plt.close(fig)


