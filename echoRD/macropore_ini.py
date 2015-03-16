# coding=utf-8

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.ndimage as snd
from scipy import ndimage
from skimage import img_as_float
from skimage.filter import sobel
from skimage.feature import peak_local_max
from skimage import morphology
from skimage import measure
from skimage.color import label2rgb
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from descartes.patch import PolygonPatch
from shapely.geometry import asMultiPoint

#THIS FUNCTION READS PREPARED HORIZONTAL IMAGES TO SETUP THE MACROPORE DOMAIN FROM BRILLIANT BLUE STAINED EXPERIMENTS
def macfind_g(fi,patch_threshold):
    # fi is file to read
    # patch_threshold are min [0] and max [1] of the desired patch size limits
    # read prepared horizontal BB image (res: 1px=1mm)
    # convert to float numbers
    Lface=snd.imread(fi)
    im = img_as_float(Lface)
    sim=np.shape(im)
    # calculate difference of channels to extract blue stained patches
    dim=abs(im[:,:,1]-im[:,:,0])
    # discard low contrasts
    dim[dim<0.3]=0.0

    # filter to local maxima for further segmentation
    # process segmentation according to sobel function of skimage
    # patch_threshold = 51 #define theshold for macropore identification
    image_max = ndimage.maximum_filter(dim, size=5, mode='constant')

    elevation_map = sobel(dim)

    markers = np.zeros_like(dim)
    markers[image_max < 0.1] = 2
    markers[image_max > 0.2] = 1

    segmentation = morphology.watershed(elevation_map, markers)

    segmentation = ndimage.binary_fill_holes(1-(segmentation-1))

    # clean patches below theshold
    patches_cleaned = morphology.remove_small_objects(segmentation, patch_threshold[0])
    labeled_patches, lab_num = ndimage.label(patches_cleaned)
    sizes = np.bincount(labeled_patches.ravel())[1:] #first entry (background) discarded
    
    # reanalyse for large patches and break them by means of watershed segmentation
    idx=np.where(sizes>patch_threshold[1])[0]+1
    labeled_patches_large=labeled_patches*0
    idy=np.in1d(labeled_patches,idx).reshape(np.shape(labeled_patches))
    labeled_patches_large[idy]=labeled_patches[idy]
    distance = ndimage.distance_transform_edt(labeled_patches_large)
    footp=int(np.round(np.sqrt(patch_threshold[1])/100)*100)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((footp, footp)),labels=labeled_patches_large)
    markers = ndimage.label(local_maxi)[0]
    labels_broken_large = morphology.watershed(-distance, markers, mask=labeled_patches_large)
    labeled_patches[idy]=labels_broken_large[idy]+np.max(labeled_patches)
    ulabels=np.unique(labeled_patches)[1:]
    sizes = np.bincount(labeled_patches.ravel())[1:]
    sizes = sizes[ulabels-1]
    inlabels=ulabels-1

    # measures
    meas=measure.regionprops(labeled_patches, properties=['Area', 'Centroid'], intensity_image=None)
    
    centroidx=ulabels.astype(np.float64)
    centroidy=ulabels.astype(np.float64)
    filledarea=ulabels.astype(np.float64)
    perimeter=ulabels.astype(np.float64)
    diameter=ulabels.astype(np.float64)

    for i in np.arange(len(ulabels)):
        ix=inlabels[i]
        centroidx[i],centroidy[i]=meas[ix]['Centroid']
        filledarea[i]=meas[ix]['FilledArea']
        perimeter[i]=meas[ix]['Perimeter']
        diameter[i]=meas[ix]['EquivDiameter']
        
    #calculate min/max distances of centroids
    mindist=ulabels.astype(np.float64)
    maxdist=ulabels.astype(np.float64)
    meandist=ulabels.astype(np.float64)
    mediandist=ulabels.astype(np.float64)

    for i in np.arange(len(ulabels)):
        cxm = np.ma.array(np.append(centroidx,[0.1*sim[0],0.9*sim[0],0.1*sim[0],0.9*sim[0]]), mask=False)
        cym = np.ma.array(np.append(centroidy,[0.1*sim[1],0.1*sim[1],0.9*sim[1],0.9*sim[1]]), mask=False)
        cxm.mask[i] = True
        cym.mask[i] = True
        mindist[i]=np.sqrt(np.min((cxm - centroidx[i])**2 + (cym - centroidy[i])**2))
        maxdist[i]=np.sqrt(np.max((cxm - centroidx[i])**2 + (cym - centroidy[i])**2))
        meandist[i]=np.mean(np.sqrt((cxm - centroidx[i])**2 + (cym - centroidy[i])**2))
        mediandist[i]=np.median(np.sqrt((cxm - centroidx[i])**2 + (cym - centroidy[i])**2))
    
    inan=-np.isnan(mediandist)
    tot_size=np.float64(im.shape[0])*np.float64(im.shape[1])
    if (len(filledarea)>0):
        patch_def=pd.DataFrame([dict(no=len(ulabels), share=np.sum(filledarea)/tot_size, 
                                 minA=np.min(filledarea), maxA=np.max(filledarea), meanA=np.mean(filledarea), medianA=np.median(filledarea),
                                 minP=np.min(perimeter), maxP=np.max(perimeter), meanP=np.mean(perimeter), medianP=np.median(perimeter),
                                 minDia=np.min(diameter), maxDia=np.max(diameter), meanDia=np.mean(diameter), medianDia=np.median(diameter),
                                 minmnD=np.min(mindist), maxmnD=np.max(mindist), meanmnD=np.mean(mindist), medianmnD=np.median(mindist),
                                 minmxD=np.min(maxdist), maxmxD=np.max(maxdist), meanmxD=np.mean(maxdist), medianmxD=np.median(maxdist),
                                 minmD=np.min(meandist), maxmD=np.max(meandist), meanmD=np.mean(meandist),
                                 minmdD=np.min(mediandist[inan]), maxmdD=np.max(mediandist[inan]), meanmdD=np.mean(mediandist[inan]), 
                                 skewD=np.mean(mediandist[inan]-meandist[inan]), skewmxD=np.max(mediandist[inan]-meandist[inan])), ])
    else:
        patch_def=[]      
    
    image_label_overlay = label2rgb(labeled_patches, image=image_max)

    # plot results
    #plt.figure(figsize=(10, 5))
    plt.subplot(132)
    plt.imshow(image_max, cmap=plt.cm.gray, interpolation='nearest')
    plt.contour(labeled_patches, [0.5], linewidths=1.2, colors='y')
    plt.axis('off')
    plt.title('identified patches')
    plt.subplot(133)
    plt.imshow(image_label_overlay, interpolation='nearest')
    plt.axis('off')
    plt.title('labeled patches')
    plt.subplot(131)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('input image')

    #plt.subplots_adjust(**margins)
    plt.show()
    
    return patch_def

#NOTE:
#Stained patches are seen as areas with macropore matrix interaction. We do not assume any tree network of pores explicitly. However, any patch must be connected to the surface. One could now analyse the overlaying matches of identified pores to get a more clear image. We will simplify this to the assumption, that the number of pores will not increase with depth and that non-visible paths will simply get a very low contact interface where they were not visible.
#At the same time we do not necessarily need to take all the macropores to the model domain. Although the more representatives we choose the better the result may become, a minimal representative set is defined by the rarest macropore class.


def mac_matrix_setup(mac,mc):
    # Find lowest number of stained patches
    #idx=np.argmin(mac.no)
    # or take largest distance
    idx=np.argmax(mac.medianmnD)
    if np.isnan(idx):
        scalingfac=1.
        scaled_no=np.array([1])
    else:
        # get scaling factor
        scalingfac=mac.no[idx]
        # define transfer macropores
        scalingfac=min(mac.no/scalingfac.astype(np.float64))*scalingfac #at least 1 pore in each layer
        scaled_no=np.round(mac.no/scalingfac) #scaled number of macropores

    # make monotonously decreasing over depth by adding transfer pores
    transferpore=mac.no*0
    for i in np.arange(len(mac)-1)[::-1]:
        transferpore[i]=max(0,(transferpore[i+1]+scaled_no[i+1])-scaled_no[i])
    
    # Add assumption for deep draining macropores
    # we need at least maxdepth_mac, perimeter
    maxdepth_mac=-mc.soildepth
    perimeter=30. # [mm]
    contact_fac=5.
    dia_transfer=5.
    perimeter_transfer=2*np.pi*dia_transfer  # [mm]
    #DEBUG: Make parameters explicit!

    # setup macropore AND matrix domain
    # take maximum of median of observed min distances to span domain
    if mc.nomac==False:
        domain_width=max(scaled_no*mac.medianmnD)/1000. #convert mm to m
    elif mc.nomac==True:
        domain_width=0.01
    else:
        domain_width=mc.nomac


    # distribute macropores
    tot_pores=transferpore[0]+scaled_no[0]
    md_pos=np.zeros(tot_pores)
    md_dia=np.zeros((tot_pores, len(mac)+1))
    md_contact=np.zeros((tot_pores, len(mac)+1))
    md_area=np.zeros((tot_pores, len(mac)+1))
    md_share=np.zeros((tot_pores, len(mac)+1))
    md_depth=np.append(mac.depth.values,maxdepth_mac) #macropore depth as observed

    # get first pore
    md_pos[0]=np.random.rand()*domain_width #lateral position of first macropore at random
    md_dia[0,:]=0.004
    md_contact[0,:]=(perimeter/contact_fac)/1000.
    md_area[0,:]=(perimeter/contact_fac)/1000.

    # add pore defs according to prepared distributions
    for i in np.arange(len(mac))[::-1]:
        #md_pos=md_pos[len(mac)-1-i]+#others follow with minimum distance according to databasis
        RR=np.random.rand(scaled_no[i],3)*2-1
        dummy=((mac.medianmnD[i]+RR[:,0]*(min(mac.medianmnD[i]-mac.minmnD[i],mac.maxmnD[i]-mac.medianmnD[i])))*mc.stain_res)
        md_pos[np.arange(int(scaled_no[i]))]=np.repeat(dummy, i+1).reshape((int(scaled_no[i]),i+1))
        dummy=((mac.medianDia[i]+RR[:,1]*(min(mac.medianDia[i]-mac.minDia[i],mac.maxDia[i]-mac.medianDia[i])))*mc.stain_res)
        md_dia[np.arange(int(scaled_no[i])),0:i+1]=np.repeat(dummy, i+1).reshape((int(scaled_no[i]),i+1))
        dummy=((mac.medianP[i]+RR[:,2]*(min(mac.medianP[i]-mac.minP[i],mac.maxP[i]-mac.medianP[i])))/contact_fac)*mc.stain_res
        md_contact[np.arange(int(scaled_no[i])),0:i+1]=np.repeat(dummy, i+1).reshape((int(scaled_no[i]),i+1))
        dummy=(mac.medianA[i]+RR[:,2]*(min(mac.medianA[i]-mac.minA[i],mac.maxA[i]-mac.medianA[i])))*mc.stain_res*mc.stain_res
        md_area[np.arange(int(scaled_no[i])),0:i+1]=np.repeat(dummy, i+1).reshape((int(scaled_no[i]),i+1))
        
        md_dia[int(scaled_no[i]-1):int(scaled_no[i]-1+transferpore[i]),0:i+1]=dia_transfer*mc.stain_res
        md_contact[int(scaled_no[i]-1):int(scaled_no[i]-1+transferpore[i]),0:i+1]=(perimeter_transfer/contact_fac)*mc.stain_res

    #in case of single mac:
    if mc.nomac==domain_width:
        md_pos[0]=0.5*domain_width #center macropore
        
    # one may use md_contact.T and md_dia.T for easier visualisation
    # check for domain bounds
    md_pos=np.cumsum(md_pos)
    md_pos[(md_pos>domain_width) | (md_pos<0)]=abs(md_pos[(md_pos>domain_width) | (md_pos<0)]-domain_width)
    md_macdepth=md_depth[np.sum(np.ceil(md_contact),axis=1).astype(int)-1] #max depth of each macropore

    #Reconsider how to handle particles in macropores and matrix domain and how they can be addressed quickly...
    #One may either construct some polygon interface area for macropores and check for belongig to area -- or use a grid, which will be in place anyways...
    #Two step strategy: First assumption on grid. Second considers contact polygon.

    #pd.DataFrame(md_contact.T, columns=md_pos, index=md_depth)
    # add lateral dimension to macropores in terms of total capacity
    #pd.DataFrame(md_area.T, columns=md_pos, index=md_depth)
    # >> depends on particle size definition :: handled there

    # create contact polygon
    xleft=np.repeat(md_pos,len(md_depth)).reshape(md_contact.shape)-md_contact/2
    xright=np.repeat(md_pos,len(md_depth)).reshape(md_contact.shape)+md_contact/2
    # add surface vertex
    xleft=np.vstack((xleft[:,0],xleft.T)).T
    xright=np.vstack((xright[:,0],xright.T)).T
    md_depth=np.append(0.0,md_depth)

    macP=MultiPolygon([ Polygon(zip(np.concatenate((xleft[i,],xright[i,::-1]),axis=1),np.concatenate((-1*md_depth,-1*md_depth[::-1]),axis=1))) for i in np.arange(len(xleft))])

    #NOTE:
    #a) reconsider if macropores may get less contact interface and be more sharply defined > matter of parameterisation
    #b) clip polygons if overlapping

    # use a +/- regular grid instead with easy conversion of position to index
    # identify which grid cells intersect with macropore interface

    ## span matrix domain and define grid
    domain_depth=mc.soildepth
    latgrid=round(domain_width/mc.grid_sizefac)
    latfac=domain_width/latgrid #factor to quickly convert position to grid
    vertgrid=round(abs(domain_depth)/mc.grid_sizefac)
    vertfac=domain_depth/vertgrid #factor to quickly convert position to grid

    # get index matrices
    # soil type, contact to macropore...
    # handle as maskes...
    mxdomain=np.zeros((latgrid,vertgrid))
    mxsoil=np.zeros((vertgrid), dtype=int)

    #SOIL MATRIX DEFINITIONS
    matrixdef=pd.read_csv(mc.matrixdeffi, sep=',')
    # find overlapping definitions
    vertices=np.unique(np.append(matrixdef.zmax,matrixdef.zmin))
    # refer soil defs to vertices
    for i in np.arange(len(vertices[:-1])):
        idx=matrixdef.no[(matrixdef['zmax']>vertices[i]) & (matrixdef['zmin']<=vertices[i])]
        idz=abs(np.arange(int(round(vertices[i+1]/vertfac)),int(round(vertices[i]/vertfac))))
        idz=idz[idz<vertgrid]
        if len(idx)<2:
            mxsoil[idz]=idx
        else:
            dummy=np.random.rand(len(idz))
            dummi=np.arange(len(idz))*0
            ratio=np.cumsum(matrixdef.trust.iloc[idx.index].values/sum(matrixdef.trust.iloc[idx.index].values))
            for k in np.arange(len(idx)-1):
                dummi[dummy>ratio[k]]=k+1
            mxsoil[idz]=idx.values[dummi]

    soilgrid=mxsoil.repeat(latgrid).reshape(vertgrid,latgrid)
    # imprint macropore connections
    # the centroid of each grid node decides if a grid is connected to a macropore or not

    # refer macropore defs to vertices
    macgrid=np.zeros((vertgrid), dtype=int)
    idx=abs(md_depth[:-1]/vertfac).astype(int)
    for i in np.arange(len(idx)):
        macgrid[idx[i]:]=i

    # setup grid centroids
    dummyl=latfac*(1+np.arange(latgrid))-latfac/2.
    dummyv=vertfac*(1+np.arange(vertgrid))-vertfac/2.

    gridcentroids = MultiPoint(zip(np.repeat(dummyl,vertgrid).reshape(latgrid,vertgrid).ravel(),np.repeat(dummyv,latgrid).reshape(vertgrid,latgrid).T.ravel()))
    onepartpercell = np.repeat(dummyl,vertgrid).reshape(latgrid,vertgrid).ravel(),np.repeat(dummyv,latgrid).reshape(vertgrid,latgrid).T.ravel()
    mxdepth_cx=dummyv.repeat(latgrid).reshape(vertgrid,latgrid)
    mxdepth_cr=mxdepth_cx.ravel()

    # loop through domain and check for overlap
    # give 0 for no or i for i'th macropore
    macconnect=np.zeros(len(gridcentroids),dtype=int)
    if mc.nomac!=True:
        for i in np.arange(len(gridcentroids)):
            if gridcentroids[i].intersects(macP[0]):
                macconnect[i]=1
            for j in np.arange(len(macP)-1)+1:
                if gridcentroids[i].intersects(macP[j]):
                    macconnect[i]=j+1

    macconnect=macconnect.reshape(latgrid,vertgrid).T

    #macconnect.shape
    macid=[]
    for i in np.arange(len(macP)):
        macid.append(np.where(macconnect.ravel()==i+1))

    #define macropore centroid index vector
    row_dummy=np.repeat(np.arange(vertgrid),len(md_pos))
    col_dummy=np.repeat(np.floor(md_pos/latfac).astype(int),vertgrid)
    mac_cells=(row_dummy*latgrid + col_dummy).astype(np.int64)

    #the domain is set. however, consider much more srictly defining the macropore contact areas (DEBUG)
    #we may still need more preps of input data etc.
    #...this module needs still some cleaning.
    mgrid=pd.DataFrame(np.array((vertgrid,latgrid,vertfac,latfac,domain_width,domain_depth)),index=['vertgrid', 'latgrid', 'vertfac', 'latfac', 'width', 'depth']).T
    mc.md_contact=md_contact #pd.DataFrame(md_contact.T, columns=md_pos, index=md_depth)
    mc.md_area=md_area#pd.DataFrame(md_area.T, columns=md_pos, index=md_depth)
    mc.md_pos=md_pos
    mc.md_share=mac.share.values
    mc.md_macdepth=md_macdepth
    mc.maccols=np.floor(mc.md_pos/mgrid.latfac.values).astype(np.int)
    mc.soilgrid=soilgrid
    mc.macconnect=macconnect
    mc.maccells=mac_cells
    mc.matrixdef=matrixdef
    mc.mgrid=mgrid
    mc.macid=macid
    mc.macP=macP
    mc.md_depth=md_depth
    mc.mxdepth_cr=mxdepth_cr
    mc.onepartpercell=onepartpercell
    mc.zgrid=(np.arange(0,mc.mgrid.depth.values,mc.mgrid.vertfac.values,dtype=np.float)+(mc.mgrid.vertfac.values/2.)).repeat(mc.mgrid.latgrid.values.astype(int)[0]).reshape(mc.mgrid.vertgrid.values.astype(int)[0],mc.mgrid.latgrid.values.astype(int)[0])

    return mc
    #macP,macid,macconnect,soilgrid,matrixdef,mgrid went obsolete to pass since they reside in mc




def mac_plot(macP):
    '''Plot Macropore Interface Setup based on MultiPolygon
    '''
    COLORs = ['#FF6347','#FF8C00','#9ACD32','#20B2AA','#1E90FF','#9932CC','#A0522D','#EF471A','#10A325','#CCD315','#FF6347','#FF8C00','#9ACD32','#20B2AA','#1E90FF','#9932CC','#A0522D','#EF471A','#10A325','#CCD315']

    def v_color(ob):
        return COLOR[ob.is_valid]

    def plot_coords(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, 'o', color='#999999', zorder=1)

    # plot
    ax = plt.figure(figsize=(8, 4))
    ax = plt.subplot(121)
    for i in np.arange(len(macP)):
        plot_coords(ax, macP[i].exterior)
        patch = PolygonPatch(macP[i], facecolor=COLORs[i], edgecolor=COLORs[i], alpha=0.5, zorder=2)
        ax.add_patch(patch)
        
    ax.set_title('Macropore Setup')
    plt.show()

