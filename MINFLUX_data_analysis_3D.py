# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:12:54 2024
Version 1

Programmed in Python 3.7 enviroment. Should be upwards compatible.
@author: OttoWirth Nikunj

Use MINFLUX_data_analysis_3D for going thorugh all 3D data with .npy format. 
All the data with .npy files are in MINFLUX Data NPY format->2D data. All the csv files will be stored in 2D data and all the figs in 2D Figs as file name_figX. Copy all the csv files to 2D CSV folder and then run combinedCSV.py
"""

#load packages needed for script
import tkinter as tk
import tkinter.filedialog as fd
from matplotlib import pyplot as plt
from IPython import get_ipython
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
import os

#%% USER INPUT REQUIRED HERE

# --- Define thresholds! Adjust these to your data! ---
efo_max = 200*1000  #max efo in Hz. this varies between 100-250 kHz
cfr_max = 0.9       #max cfr. Optimize as per experiment
len_min = 3         #min number of localizations per tid. Optimize as per experiment
save_results = True #set True to save results as csv

# --- User input to select data file ---
root = tk.Tk()
root.withdraw() # Hide the main tkinter window
data_path = fd.askopenfilename(parent=root, title='Choose the MINFLUX data file (.npy)')

output_folder = ''    # <- Update this. Folder location for saving Figures
folder_name1 = ''    # <- Update this. Folder location for saving NND graphs
folder_name2 = ''    # <- Update this. Folder location for saving All results
folder_name3 = ''    # <- Update this. Folder location for saving syt1 results
folder_name4 = ''    # <- Update this. Folder location for saving syt7 results

# Extract the base file name without extension for naming output files
base_name = os.path.splitext(os.path.basename(data_path))[0]
output_file1 = f"{base_name}_Fig1.png"
output_file2 = f"{base_name}_Fig2.png"
output_file3 = f"{base_name}_Fig3.png"
output_file1_pdf = f"{base_name}_Fig1.pdf"
output_file2_pdf = f"{base_name}_Fig2.pdf"
output_file3_pdf = f"{base_name}_Fig3.pdf"

# Create full paths for the output figures
output_fig1 = os.path.join(output_folder, output_file1)
output_fig2 = os.path.join(output_folder, output_file2)
output_fig3 = os.path.join(output_folder, output_file3)
output_fig1_pdf = os.path.join(output_folder, output_file1_pdf)
output_fig2_pdf = os.path.join(output_folder, output_file2_pdf)
output_fig3_pdf = os.path.join(output_folder, output_file3_pdf)


#%% FUNCTIONS
#define functions used in script
def get_relevant_MFX_data(mfx_path):
    """Loads and pre-processes MINFLUX data from a .npy file."""
    mfx=np.load(mfx_path)
    loc_id = np.cumsum(np.append(0,np.diff(mfx['itr'])<1))
    mfx['fnl'] = np.logical_and(mfx['itr']==max(mfx['itr']), mfx['vld'])
    mfx = mfx[np.any(loc_id==loc_id[mfx['fnl'],None],axis=0)] #remove non-complete localizations
    cfr_id = max(mfx['itr'][mfx['cfr']>0]) #identify last iteration with cfr check
    mfx['cfr'][mfx['fnl']] = mfx['cfr'][mfx['itr']==cfr_id]
    repItrDcr = np.reshape(mfx['dcr'][mfx['itr']>=cfr_id,0],(mfx['fnl'].sum(),max(mfx['itr'])-cfr_id+1)) #DCR including headstart iteration
    repItrEco = np.reshape(mfx['eco'][mfx['itr']>=cfr_id],(mfx['fnl'].sum(),max(mfx['itr'])-cfr_id+1)) #EFO including headstart iteration
    weighted_avg = np.average(repItrDcr, axis = 1, weights=repItrEco) #iteration averaged DCR
    mfx['dcr'][mfx['fnl'],0] = weighted_avg
    
    return mfx[mfx['fnl']]
    
def grp_mean(arr,grp): #function to calculate the mean of any array according to the group index
    u_arr, idx, count  = np.unique(grp,return_index=True,return_counts=True)
    grpm_arr = np.add.reduceat(arr, idx, axis=0) / count[:,None]
    return grpm_arr, u_arr, count

def colorSeparation(mfx): #improved color separation
    """Separates localizations into two color channels based on DCR values using a Gaussian Mixture Model."""
    gm = GaussianMixture(n_components=2, random_state=0).fit(mfx['dcr'][:,0].reshape(-1, 1)) #Apply Gaussian mixture model (GMM) to find DCR peaks
    #get localizations probability of belonging to either color channel
    comp2=np.argmax(gm.means_)
    gmprob = gm.predict_proba(mfx['dcr'][:,0].reshape(-1, 1))
    #assign label according to calcualted probability (threshold 95%)
    colorlabel = np.zeros(mfx['dcr'][:,0].reshape(-1, 1).size)
    colorlabel[gmprob[:,comp2]>0.95] = 1
    colorlabel[(gmprob[:,0]<0.95) & (gmprob[:,1]<0.95)] = 2

    #assign label according to majority in trace
    for tid in np.unique(mfx['tid']):
        uv,c= np.unique(colorlabel[mfx['tid']==tid], return_counts=True)
        colorlabel[mfx['tid']==tid] = uv[np.argmax(c)]
    
    return colorlabel
#%% MAIN SCRIPT

#load data
mfx = get_relevant_MFX_data(data_path)  #get relveant mfx data

#% %FILTER DATA

#plot histograms
get_ipython().run_line_magic('matplotlib', 'qt')    #make plots appear as separate windows (also closes all current plots)

plt.figure(layout="constrained")
plt.subplot(1,3,1)
H=plt.hist(mfx['efo']/1e3,bins=32,label='Data')
plt.plot([efo_max/1e3,efo_max/1e3],[0,max(H[0])],label='EFO max')
plt.xlabel('EFO (kHz)')
plt.ylabel('Occurrence')
plt.title(f'efo_max={efo_max/1000}k')
plt.xlim(0, 500) # Set x-axis limit to 0-500 kHz (0-500,000 Hz)
plt.legend()

plt.subplot(1,3,2)
H=plt.hist(mfx['cfr'],bins=32,label='Data')
plt.plot([cfr_max,cfr_max],[0,max(H[0])],label='CFR max')
plt.xlabel('CFR')
plt.ylabel('Occurrence')
plt.title(f'cfr_max={cfr_max}')
plt.legend()

#remove localizations based on efo and cfr
mfx = mfx[(mfx['efo']<efo_max) & (mfx['cfr']<cfr_max)]

#extract number of localizations per tid
tid_uniq, tid_inv, tid_freq = np.unique(mfx['tid'] , return_counts=True, return_inverse=True)

#plot histogram for number of localizations per TID
plt.subplot(1,3,3)
H=plt.hist(tid_freq,bins=32,label='Data')
plt.plot([len_min,len_min],[0,max(H[0])],label='min loc per tid')
plt.xlabel('Localizations per tid')
plt.ylabel('Occurrence')
plt.legend()
plt.savefig(output_fig1)
plt.savefig(output_fig1_pdf)

#remove localizations based on number of localizations per tid
mfx = mfx[~(tid_freq[tid_inv]<len_min)]

#%% improved color separation

colorlabel = colorSeparation(mfx) #performs color separation

#Plot histograms of DCR and assigned DCR
plt.figure(layout="constrained")
plt.subplot(1,2,1)
plt.hist(mfx['dcr'][:,0], bins=32, range=(0,1))
plt.xlabel('DCR')
plt.ylabel('Occurance')
plt.title('Eco averaged DCR')
plt.show()

plt.subplot(1,2,2)
plt.hist(mfx['dcr'][:,0][colorlabel==0], bins=30, range=(0,1), alpha = 0.5, label = 'ch 1')
plt.hist(mfx['dcr'][:,0][colorlabel==1], bins=30, range=(0,1), alpha = 0.5, label = 'ch 2') 
plt.legend()
plt.xlabel('DCR')
plt.ylabel('Occurance')
plt.title('Automatic DCR two colors sep')
plt.savefig(output_fig2)
plt.savefig(output_fig2_pdf)
plt.show()

#plot data as xy scatter
plt.figure(figsize=(15,9))
plt.subplot(1,2,1)
plt.scatter(mfx['loc'][colorlabel==0,0],mfx['loc'][colorlabel==0,1],color='m',label='680nm')
plt.scatter(mfx['loc'][colorlabel==1,0],mfx['loc'][colorlabel==1,1],color='c',label='640nm')
plt.gca().set_aspect('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()

#%% calculate NNDistance

#calculate TID center
_data = np.hstack((mfx['loc'],
                   mfx['efo'][:,None],
                   mfx['cfr'][:,None],
                   mfx['dcr'][:,0][:,None],
                   mfx['tim'][:,None],
                   mfx['tid'][:,None],
                   np.zeros((len(mfx),1)),
                   colorlabel[:,None])) #create array with columns = x,y,z,color
meanLoc = grp_mean(_data,mfx['tid']) #calculate the mean value of each column for all values with the same TID
loc = meanLoc[0][:,0:3]
clabel = meanLoc[0][:,-1]

#initialize arrays for NND
NND_680to640 = np.array([])
NND_640to680 = np.array([])

# Calculate NND only if both colors are present
if (np.any(clabel==0)) and (np.any(clabel==1)):
    loc_c0 = loc[clabel==0]
    loc_c1 = loc[clabel==1]
    
    NND_680to640=np.zeros(len(loc_c0))
    NND_640to680=np.zeros(len(loc_c1))
    
    for i, p0 in enumerate(loc_c0): #calculate nearest-neighbor distance 680 to 640
        NND_680to640[i] = np.min(np.sqrt(np.sum((p0 - loc_c1)**2, axis=1)))
        
    for i, p1 in enumerate(loc_c1): #calculate nearest-neighbor distance 640 to 680
        NND_640to680[i] = np.min(np.sqrt(np.sum((p1 - loc_c0)**2, axis=1)))

    meanLoc[0][clabel==0,-2] = NND_680to640 #680 to 640
    meanLoc[0][clabel==1,-2] = NND_640to680 #640 to 680
    if np.any(clabel==2):
        meanLoc[0][clabel==2,-2] = np.nan #non categorized to 680 or 640
        
    #plot histogram of NND
    plt.subplot(1,2,2)
    plt.hist(NND_680to640*1e9,bins=32,range=(0,200),histtype='step',color='red',label='680 to 640')
    plt.hist(NND_640to680*1e9,bins=32,range=(0,200),histtype='step',color='green',label='640 to 680')
    plt.xlabel('NND (nm)')
    plt.ylabel('Occurrence')
    plt.legend()
    plt.savefig(output_fig3)
    plt.savefig(output_fig3_pdf)
else:
    meanLoc[0][:,-2] = np.nan # Set NND to NaN if one or both colors are missing
    print("Could not calculate NND: one or both color channels are empty after filtering.")


#%% save results as csv file for batch analysis
if save_results and NND_680to640.size > 0 and NND_640to680.size > 0:
    max_len = max(len(NND_680to640), len(NND_640to680))
    df = pd.DataFrame(columns=['NND_680to640', 'NND_640to680'], index=range(max_len))
    
    # Pad the shorter array with NaNs to make them equal length for the DataFrame
    df.NND_680to640 = pd.Series(NND_680to640 * 1e9)
    df.NND_640to680 = pd.Series(NND_640to680 * 1e9)
    
    df.to_csv(os.path.join(folder_name1, base_name + '_NND.csv'), header=True, index=False)
elif save_results:
    print("Skipping NND CSV save because one or both channels were empty.")

#%% save detailed results as csv file for batch analysis
    
csv_fmt_specifier = ['%.13f', '%.13f', '%.13f', '%f', '%f', '%f', '%f', '%.0f', '%f', '%.0f']
    
if save_results:
    header_str = 'X [um],Y [um],Z [um],EFO [Hz],CFR,DCR,Time [s],TID,NND_C1toC2 [um],ColorLabel'
    np.savetxt(os.path.join(folder_name2, base_name + '_Results.csv'), meanLoc[0], delimiter = ',',
               comments = '', fmt = csv_fmt_specifier, 
               header = header_str)    
    
    meanLoc_array = meanLoc[0]    
    meanLoc_array_syt1 = meanLoc_array[meanLoc_array[:, 9] == 1]
    meanLoc_array_syt7 = meanLoc_array[meanLoc_array[:, 9] == 0]
    
    # Save results for color channel 1 (syt1)
    if meanLoc_array_syt1.size > 0:
        np.savetxt(os.path.join(folder_name3, base_name + '_Results_syt1.csv'), meanLoc_array_syt1, delimiter = ',',
                   comments = '', fmt = csv_fmt_specifier, 
                   header = header_str.replace('ColorLabel', 'ColorLabel1'))
    
    # Save results for color channel 0 (syt7)
    if meanLoc_array_syt7.size > 0:
        np.savetxt(os.path.join(folder_name4, base_name + '_Results_syt7.csv'), meanLoc_array_syt7, delimiter = ',',
                   comments = '', fmt = csv_fmt_specifier, 
                   header = header_str.replace('ColorLabel', 'ColorLabel0'))

print(f"\nAnalysis complete. Results saved in: {output_base_dir}")
