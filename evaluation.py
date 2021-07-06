




## evaluate accuracy for # of slices
from diceCoeff import diceCoeff
slices = [15, 20,25]

# print('First model results:')
# for ss in slices:
#     print('{0} slices.'.format(ss))
#     for id in range(0,len(subjs['train'])):
#         print("Train data:")
#         nii = nib.load(initoutputs_for_refstage[id])
#         gt = nib.load(labels[id])
#         mask = nib.load(masks[id])
#         data = nii.get_fdata(nii)
#         datagt = nii.get_fdata(gt)
#         datamask = nii.get_fdata(mask)
#         data[datamask ==0] =0
#         datagt[datamask == 0] = 0
#         print(initoutputs_for_refstage[id])
#         dc_wm = diceCoeff(data, datagt, 1)
#         dc_gm = diceCoeff(data, datagt, 2)
#         dc_csf = diceCoeff(data, datagt, 3)
#         print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))
#     for id in range(0,len(subjs['val'])):
#         print("Validation data:")
#         nii = nib.load(valinitoutputs_for_refstage[id])
#         gt = nib.load(vallabels[id])
#         mask = nib.load(valmasks[id])
#         data = nii.get_fdata(nii)
#         datagt = nii.get_fdata(gt)
#         datamask = nii.get_fdata(mask)
#         data[datamask ==0] =0
#         datagt[datamask == 0] = 0
#         print(valinitoutputs_for_refstage[id])
#         dc_wm = diceCoeff(data, datagt, 1)
#         dc_gm = diceCoeff(data, datagt, 2)
#         dc_csf = diceCoeff(data, datagt, 3)
#         print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))

print('Second model results:')
for ss in slices:
    print('{0} slices.'.format(ss))
    suffix = '2021_{0}slices'.format(ss) + str(0)
    refineoutputs = []
    for i in np.arange(3):
        refineoutput = '//data/infant/outputs/{4}_{0}ch_refineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(
            numchannels, epoch, i, 0,
            subjs['train'][i], suffix)
        refineoutputs.append(refineoutput)
    valrefineoutputs = []
    for i in np.arange(3):
        valrefineoutput = '//data/infant/outputs/{4}_{0}ch_valrefineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(
            numchannels, epoch, i, 0,
            subjs['val'][i], suffix)
        valrefineoutputs.append(valrefineoutput)
    # for id in range(0,len(subjs['train'])):
    #     print("Train data:")
    #     nii = nib.load(refineoutputs[id])
    #     gt = nib.load(labels[id])
    #     mask = nib.load(masks[id])
    #     data = nii.get_fdata()
    #     datagt = gt.get_fdata()
    #     datamask = mask.get_fdata()
    #     data[datamask ==0] =0
    #     datagt[datamask == 0] = 0
    #     print(refineoutputs[id])
    #     dc_wm = diceCoeff(data, datagt, 1)
    #     dc_gm = diceCoeff(data, datagt, 2)
    #     dc_csf = diceCoeff(data, datagt, 3)
    #     print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))
    for id in range(0,len(subjs['val'])):
        print("Validation data:")
        nii = nib.load(valrefineoutputs[id])
        gt = nib.load(vallabels[id])
        mask = nib.load(valmasks[id])
        data = nii.get_fdata()
        datagt = gt.get_fdata()
        datamask = mask.get_fdata()
        data[datamask ==0] =0
        datagt[datamask == 0] = 0
        print(valrefineoutputs[id])
        dc_wm = diceCoeff(data, datagt, 1)
        dc_gm = diceCoeff(data, datagt, 2)
        dc_csf = diceCoeff(data, datagt, 3)
        print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))